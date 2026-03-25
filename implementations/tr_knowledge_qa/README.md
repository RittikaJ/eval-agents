# TR Knowledge QA — Evaluation

**Thomson Reuters | Agentic AI Evaluation Bootcamp**

This is a custom implementation that combines the **Knowledge QA reference implementation** with a **flexible trace-level trajectory evaluation** inspired by the Report Generation agent.

## Notebooks

| Notebook | What it does |
|----------|-------------|
| `01_dataset_and_agent.ipynb` | Explore DeepSearchQA dataset, run the agent on a single question, inspect the AgentResponse |
| `02_evaluation.ipynb` | Upload dataset subset to Langfuse, run output-level eval (Precision / Recall / F1) |
| `03_trajectory_evaluation.ipynb` | Two-pass experiment: output quality + trace-level behavior scoring |

Run them in order — `02` and `03` depend on the Langfuse dataset created in `02`.

## What's Different From the Standard Knowledge QA

The standard `knowledge_qa` implementation evaluates only the **final answer** (Precision/Recall/F1).

This implementation adds a second evaluation pass that scores **how** the agent behaved:

| Metric | Description |
|--------|-------------|
| `trace_tool_call_count` | Total tool calls made — checks agent effort |
| `trace_search_before_fetch` | Did `google_search` precede `web_fetch`? Enforces correct retrieval pattern |
| `trace_has_sources` | Did the agent cite source URLs? Checks groundedness |
| `trace_redundant_search_ratio` | Fraction of duplicate search queries — lower is better |
| `trace_latency_sec` | Total wall-clock time in seconds |

No hand-crafted ground truth needed — all trace metrics are deterministic rules on the actual agent trace.

## Configuration

Each notebook has a config block at the top:

```python
CATEGORY        = 'Finance & Economics'   # DeepSearchQA category
NUM_SAMPLES     = 5                       # Number of questions per run
DATASET_NAME    = 'TR-DeepSearchQA'       # Langfuse dataset name (shared across notebooks)
EXPERIMENT_NAME = 'tr-baseline-v1'        # Change this for each new experiment run
```

### Running Multiple Experiments

Update `EXPERIMENT_NAME` and re-run from the config cell. Each run creates a new named experiment in Langfuse — compare them side-by-side under **Datasets → TR-DeepSearchQA**.

## Prerequisites

- `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `~/.env`
- `uv sync` run from repo root
- Langfuse access: log in at [us.cloud.langfuse.com](https://us.cloud.langfuse.com) using GitHub (same account as Coder)
