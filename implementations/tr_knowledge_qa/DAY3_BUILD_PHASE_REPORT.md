# Build Phase Daily Report (Day 3)

## Project Objective
Build and evaluate a knowledge-grounded Q&A agent that answers complex, multi-hop questions by retrieving relevant information from a document corpus using a PlanReAct architecture.

## Dataset
- **Name:** DeepSearchQA
- **Size:** 896 questions across 17 categories
- **License:** Apache 2.0
- **Ground truth:** Included
- **Use case:** Multi-hop reasoning across diverse domains

## Progress and Insights (Day 2)
- Streamlit app was set up in Coder and connected to the evaluation pipeline.
- Offline evaluation was run on an approximately 100-question subset.
- Baseline F1, Precision, and Recall were captured and logged in Langfuse.
- Langfuse span tagging was integrated for planner, retrieval, and synthesis components.
- Retrieval failures were the dominant failure mode; plan decomposition quality was generally strong.
- LLM-as-judge scoring was stable at `temperature=0` with rubric-based prompting.

## Challenges Observed (Day 2)
- Streamlit integration in Coder required additional port configuration.
- LLM-as-judge latency was high on the full 896-question set; subset runs were used for faster iteration.
- Some Langfuse spans were not correctly attributed due to missing component ID tags.

## Day 2 Follow-up Actions
- Complete full offline evaluation on all 896 questions and compare against subset baseline.
- Build online Streamlit evaluation flow for thumbs up/down feedback.
- Refine judge rubric based on Day 2 spot checks.
- Run 2-3 configuration variants (for example retrieval settings and prompt style) and compare metrics.

## Achievements (Day 3)
- Completed offline evaluation pipeline on the full 896-question dataset.
- Finalized F1, Precision, Recall, and semantic similarity metrics in Langfuse.
- Streamlit online evaluation UI is functional and supports thumbs up/down feedback.
- Compared 2-3 agent variants with measurable metric improvements logged.
- Evaluation schematic diagram was completed and pushed to the repository.

## Challenges (Day 3)
- Increasing retrieval depth improved recall but added latency.
- Streamlit session-state handling needed extra care for multi-turn interactions.
- Minor score inconsistency across judge re-runs was addressed with `temperature=0` and a tighter rubric.

## What Worked Best
PlanReAct with rubric-based LLM-as-judge scoring (`temperature=0`) and Langfuse span-level tracing gave the most reliable and interpretable results. Retrieval top-k around 5 with semantic similarity scoring gave the best practical precision/recall balance across categories.

## Future Considerations and Lessons
- Expand online evaluation to gather larger-scale human feedback.
- Explore judge-model tuning on Thomson Reuters domain data to reduce bias.
- Catch multi-hop errors earlier by inspecting plan decomposition and planner traces first.
- Keep evaluation parallelized (`max_concurrency=5`) to reduce iteration time.
- Continue using Langfuse span attribution to separate retrieval-side vs synthesis-side failures.

