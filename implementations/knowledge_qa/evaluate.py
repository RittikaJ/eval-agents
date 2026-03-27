from __future__ import annotations

"""Evaluate the Knowledge Agent using Langfuse experiments.

This script runs the Knowledge Agent against a Langfuse dataset and evaluates
results using the DeepSearchQA LLM-as-judge methodology. Results are automatically
logged to Langfuse for analysis and comparison.

Optionally, trace-level groundedness evaluation can be enabled to check if agent
outputs are supported by tool observations.

Usage:
    # Run a full evaluation
    python evaluate.py

    # Run with custom dataset and experiment name
    python evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"

    # Enable trace groundedness evaluation
    ENABLE_TRACE_GROUNDEDNESS=true python evaluate.py
"""

import asyncio
import logging
import os
from typing import Any

import click
from dotenv import load_dotenv

try:
    from aieng.agent_evals.async_client_manager import AsyncClientManager
    from aieng.agent_evals.evaluation import run_experiment, run_experiment_with_trace_evals
    from aieng.agent_evals.evaluation.graders import create_trace_groundedness_evaluator
    from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
    from aieng.agent_evals.evaluation.types import EvaluationResult
    from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
    from aieng.agent_evals.knowledge_qa.deepsearchqa_grader import DeepSearchQAResult, evaluate_deepsearchqa_async
    from aieng.agent_evals.logging_config import setup_logging
    from langfuse.experiment import Evaluation, ExperimentResult

    RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    AsyncClientManager = None  # type: ignore[assignment]
    run_experiment = None  # type: ignore[assignment]
    run_experiment_with_trace_evals = None  # type: ignore[assignment]
    create_trace_groundedness_evaluator = None  # type: ignore[assignment]
    LLMRequestConfig = None  # type: ignore[assignment]
    EvaluationResult = None  # type: ignore[assignment]
    KnowledgeGroundedAgent = None  # type: ignore[assignment]
    DeepSearchQAResult = None  # type: ignore[assignment]
    evaluate_deepsearchqa_async = None  # type: ignore[assignment]
    setup_logging = None  # type: ignore[assignment]
    Evaluation = None  # type: ignore[assignment]
    ExperimentResult = None  # type: ignore[assignment]
    RUNTIME_IMPORT_ERROR = exc


load_dotenv(verbose=True)
if setup_logging is not None:
    setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"
DEFAULT_EXPERIMENT_NAME = "Knowledge Agent Evaluation"

VALID_MILESTONES = {"day1", "day2", "day3"}

MILESTONE_EXPERIMENT_SUFFIX = {
    "day1": "Day 1",
    "day2": "Day 2",
    "day3": "Day 3",
}

# Configuration for trace groundedness evaluation
ENABLE_TRACE_GROUNDEDNESS = os.getenv("ENABLE_TRACE_GROUNDEDNESS", "false").lower() in ("true", "1", "yes")


def _require_runtime_dependencies() -> None:
    """Raise a clear error when optional eval dependencies are not installed."""
    if RUNTIME_IMPORT_ERROR is None:
        return

    raise RuntimeError(
        "Missing runtime dependencies for Knowledge QA evaluation. "
        "Install project dependencies (for example `uv sync`), then retry. "
        f"Original import error: {RUNTIME_IMPORT_ERROR}"
    )


def _coerce_milestone(value: str | None) -> str:
    """Normalize milestone value and validate supported options."""
    normalized = (value or "none").strip().lower()
    if normalized not in VALID_MILESTONES:
        valid = ", ".join(sorted(VALID_MILESTONES))
        raise ValueError(f"Invalid milestone '{value}'. Expected one of: {valid}")
    return normalized


def _build_experiment_name(base_name: str, milestone: str) -> str:
    """Append milestone suffix to experiment name when requested."""
    suffix = MILESTONE_EXPERIMENT_SUFFIX[milestone]
    if not suffix:
        return base_name
    if base_name.endswith(f"[{suffix}]"):
        return base_name
    return f"{base_name} [{suffix}]"


async def agent_task(*, item: Any, **kwargs: Any) -> str:  # noqa: ARG001
    """Run the Knowledge Agent on a dataset item.

    Parameters
    ----------
    item : Any
        The Langfuse experiment item containing the question.
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    str
        The agent's answer text. Rich execution data (plan, tool calls,
        sources, reasoning chain) is attached to the Langfuse span metadata.
    """
    _require_runtime_dependencies()
    question = item.input
    agent_model = kwargs.get("agent_model")
    enable_planning = kwargs.get("enable_planning", True)
    thinking_budget = kwargs.get("thinking_budget", 8192)
    component_tags = kwargs.get("component_tags", {})

    # Coerce dynamic values passed from CLI to safe defaults.
    try:
        thinking_budget = int(thinking_budget)
    except (TypeError, ValueError):
        thinking_budget = 8192

    enable_planning = bool(enable_planning)
    logger.info(f"Running agent on: {question[:80]}...")

    try:
        agent = KnowledgeGroundedAgent(
            model=agent_model,
            enable_planning=enable_planning,
            thinking_budget=thinking_budget,
        )  # type: ignore[call-arg]
        response = await agent.answer_async(question)
        logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")

        # Attach rich execution data to the span metadata so it's inspectable
        # in Langfuse without cluttering the output field.
        client_manager = AsyncClientManager.get_instance()
        client_manager.langfuse_client.update_current_span(
            metadata={
                **response.model_dump(exclude={"text"}),
                "component_tags": component_tags,
                "agent_variant": {
                    "model": agent.model,
                    "enable_planning": enable_planning,
                    "thinking_budget": thinking_budget,
                },
            },
        )

        return response.text
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Error: {e}"


async def deepsearchqa_evaluator(
    *,
    input: str,  # noqa: A002
    output: str,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Evaluate the agent's response using DeepSearchQA methodology.

    This evaluator uses the modern async infrastructure with shared client
    management and retry logic.

    Parameters
    ----------
    input : str
        The original question.
    output : str
        The agent's answer text.
    expected_output : str
        The ground truth answer.
    metadata : dict[str, Any] | None, optional
        Item metadata (contains answer_type).
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with F1, precision, recall, and outcome scores.
    """
    _require_runtime_dependencies()
    output_text = str(output)
    answer_type = metadata.get("answer_type", "Set Answer") if metadata else "Set Answer"

    logger.info(f"Evaluating response (answer_type: {answer_type})...")

    try:
        # Use the modern async evaluator with default config
        result = await evaluate_deepsearchqa_async(
            question=input,
            answer=output_text,
            ground_truth=expected_output,
            answer_type=answer_type,
            model_config=LLMRequestConfig(temperature=0.0),
        )

        evaluations = result.to_evaluations()
        logger.info(f"Evaluation complete: {result.outcome} (F1: {result.f1_score:.2f})")
        return evaluations

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return DeepSearchQAResult.error_evaluations(str(e))


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int = 1,
    enable_trace_groundedness: bool = False,
    *,
    milestone: str = "none",
    agent_model: str | None = None,
    enable_planning: bool = True,
    thinking_budget: int = 8192,
    component_tags: dict[str, str] | None = None,
) -> ExperimentResult | EvaluationResult:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    enable_trace_groundedness : bool, optional
        Whether to enable trace-level groundedness evaluation, by default False.
    milestone : str, optional
        Optional run milestone tag used in experiment naming (none|day2|day3).
    agent_model : str | None, optional
        Override model for KnowledgeGroundedAgent. Uses default when None.
    enable_planning : bool, optional
        Whether to enable planning in the agent, by default True.
    thinking_budget : int, optional
        Thinking budget for planning-capable models, by default 8192.
    component_tags : dict[str, str] | None, optional
        Additional tags to store in span metadata for planner/retrieval/synthesis attribution.
    """
    _require_runtime_dependencies()
    client_manager = AsyncClientManager.get_instance()
    normalized_milestone = _coerce_milestone(milestone)
    run_name = _build_experiment_name(experiment_name, normalized_milestone)
    tags = component_tags or {
        "planner": "planner",
        "retrieval": "retrieval",
        "synthesis": "synthesis",
    }

    async def configured_agent_task(*, item: Any, **kwargs: Any) -> str:
        return await agent_task(
            item=item,
            agent_model=agent_model,
            enable_planning=enable_planning,
            thinking_budget=thinking_budget,
            component_tags=tags,
            **kwargs,
        )

    try:
        logger.info(f"Starting experiment '{run_name}' on dataset '{dataset_name}'")
        logger.info(f"Max concurrency: {max_concurrency}")
        logger.info(f"Trace groundedness: {'enabled' if enable_trace_groundedness else 'disabled'}")
        logger.info(
            "Agent variant: model=%s planning=%s thinking_budget=%s milestone=%s",
            agent_model or "<default>",
            enable_planning,
            thinking_budget,
            normalized_milestone,
        )

        result: ExperimentResult | EvaluationResult
        if enable_trace_groundedness:
            # Create trace groundedness evaluator
            # Only consider web_fetch and google_search tools as evidence
            groundedness_evaluator = create_trace_groundedness_evaluator(
                name="trace_groundedness",
                model_config=LLMRequestConfig(temperature=0.0),
            )

            # Run with trace evaluations
            result = run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=run_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge and trace groundedness",
                task=configured_agent_task,
                evaluators=[deepsearchqa_evaluator],  # Item-level evaluators
                trace_evaluators=[groundedness_evaluator],  # Trace-level evaluators
                max_concurrency=max_concurrency,
            )
        else:
            # Run without trace evaluations
            result = run_experiment(
                dataset_name=dataset_name,
                name=run_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge",
                task=configured_agent_task,
                evaluators=[deepsearchqa_evaluator],
                max_concurrency=max_concurrency,
            )

        logger.info("Experiment complete!")
        # Handle both ExperimentResult and EvaluationResult
        if isinstance(result, EvaluationResult):
            # EvaluationResult from run_experiment_with_trace_evals
            logger.info(f"Results: {result.experiment}")
            if result.trace_evaluations:
                trace_evals = result.trace_evaluations
                logger.info(
                    f"Trace evaluations: {len(trace_evals.evaluations_by_trace_id)} traces, "
                    f"{len(trace_evals.skipped_trace_ids)} skipped, {len(trace_evals.failed_trace_ids)} failed"
                )
        else:
            # ExperimentResult from run_experiment
            logger.info(f"Results: {result}")

        return result

    finally:
        logger.info("Closing client manager and flushing data...")
        try:
            await client_manager.close()
            await asyncio.sleep(0.1)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--experiment-name",
    default=DEFAULT_EXPERIMENT_NAME,
    help="Name for this experiment run.",
)
@click.option(
    "--max-concurrency",
    default=1,
    type=int,
    help="Maximum concurrent agent runs (default: 1).",
)
@click.option(
    "--milestone",
    default="none",
    type=click.Choice(["none", "day2", "day3"], case_sensitive=False),
    help="Optional milestone suffix in experiment name (none|day2|day3).",
)
@click.option(
    "--agent-model",
    default=None,
    help="Optional model override for KnowledgeGroundedAgent.",
)
@click.option(
    "--disable-planning",
    is_flag=True,
    default=False,
    help="Disable built-in planning to compare PlanReAct impact.",
)
@click.option(
    "--thinking-budget",
    default=8192,
    type=int,
    help="Thinking budget for planning-capable models.",
)
@click.option(
    "--component-tag",
    "component_tags_raw",
    multiple=True,
    help="Component tag mapping in key=value form, e.g. planner=planner_v2.",
)
@click.option(
    "--enable-trace-groundedness",
    is_flag=True,
    default=ENABLE_TRACE_GROUNDEDNESS,
    help="Enable trace-level groundedness evaluation.",
)
@click.option(
    "--run-variant-sweep",
    is_flag=True,
    default=False,
    help="Run a 3-variant comparison sweep (top-level experiment names only).",
)
def cli(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int,
    milestone: str,
    agent_model: str | None,
    disable_planning: bool,
    thinking_budget: int,
    component_tags_raw: tuple[str, ...],
    enable_trace_groundedness: bool,
    run_variant_sweep: bool,
) -> None:
    """Run Knowledge Agent evaluation using Langfuse experiments."""
    _require_runtime_dependencies()

    component_tags: dict[str, str] = {}
    for item in component_tags_raw:
        if "=" not in item:
            raise click.BadParameter(
                f"Invalid --component-tag '{item}'. Expected key=value.",
                param_hint="--component-tag",
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise click.BadParameter(
                f"Invalid --component-tag '{item}'. Expected non-empty key=value.",
                param_hint="--component-tag",
            )
        component_tags[key] = value

    planning_enabled = not disable_planning

    if run_variant_sweep:
        variants = [
            (f"{experiment_name}-baseline", planning_enabled, thinking_budget),
            (f"{experiment_name}-no-planning", False, thinking_budget),
            (f"{experiment_name}-high-thinking", planning_enabled, max(thinking_budget, 12288)),
        ]

        async def _run_sweep() -> None:
            for variant_name, variant_planning, variant_thinking in variants:
                await run_evaluation(
                    dataset_name,
                    variant_name,
                    max_concurrency,
                    enable_trace_groundedness,
                    milestone=milestone,
                    agent_model=agent_model,
                    enable_planning=variant_planning,
                    thinking_budget=variant_thinking,
                    component_tags=component_tags or None,
                )

        asyncio.run(_run_sweep())
        return

    asyncio.run(
        run_evaluation(
            dataset_name,
            experiment_name,
            max_concurrency,
            enable_trace_groundedness,
            milestone=milestone,
            agent_model=agent_model,
            enable_planning=planning_enabled,
            thinking_budget=thinking_budget,
            component_tags=component_tags or None,
        )
    )


if __name__ == "__main__":
    cli()
