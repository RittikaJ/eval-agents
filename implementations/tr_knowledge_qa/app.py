import streamlit as st
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# app.py is in eval-agents/implementations/tr_knowledge_qa/
# .env is in eval-agents/
current_file = Path(__file__).resolve()
app_dir = current_file.parent  # tr_knowledge_qa/
impl_dir = app_dir.parent      # implementations/
eval_agents_dir = impl_dir.parent  # eval-agents/
env_path = eval_agents_dir / ".env"

if env_path.exists():
    load_dotenv(env_path)
    st.sidebar.success("✓ Loaded .env configuration")
else:
    st.sidebar.error("⚠️ No .env file found. Copy .env.example to .env and configure it.")
    st.sidebar.info(f"Expected at: {env_path}")

# Add the aieng module to the path
# aieng-eval-agents is in eval-agents/aieng-eval-agents/
aieng_path = Path(__file__).parent.parent.parent / "aieng-eval-agents"
if str(aieng_path) not in sys.path:
    sys.path.insert(0, str(aieng_path))

# Try to import the required classes
try:
    from aieng.agent_evals.knowledge_qa.data.deepsearchqa import DeepSearchQADataset
    from aieng.agent_evals.knowledge_qa import KnowledgeGroundedAgent
    # from aieng.agent_evals.langfuse import init_tracing
    DATASET_AVAILABLE = True
    AGENT_AVAILABLE = True

    # Initialize Langfuse tracing
    # try:
    #     init_tracing()
    # except Exception as e:
    #     st.sidebar.warning(f"Langfuse tracing not available: {e}")

except ImportError as e:
    DATASET_AVAILABLE = False
    AGENT_AVAILABLE = False
    st.error(f"""
    **Missing Dependencies**

    Please install the required packages:
    ```bash
    pip install kagglehub pandas pydantic
    ```

    Or install the full package:
    ```bash
    cd eval-agents/aieng-eval-agents
    pip install -e .
    ```

    Error: {str(e)}
    """)

# Set page configuration
st.set_page_config(
    page_title="DeepSearchQA Knowledge",
    page_icon="🔍",
    layout="wide"
)

# Main content
st.title("👋 DeepSearchQA Knowledge")

st.markdown("""
**How it works:**
1. 📚 Browse 5 random history questions from the DeepSearchQA dataset
2. 🎯 Click on a question tile to select it
3. 🤖 Click "Process with Agent" to get an AI-powered answer
4. 👍👎 Rate the agent's response to help improve the system
""")

# Load dataset and get History questions
@st.cache_data
def load_history_questions():
    """Load 5 random History questions from DeepSearchQA dataset."""
    if not DATASET_AVAILABLE:
        return []

    try:
        import pandas as pd

        # Load the local CSV file
        csv_path = Path(__file__).parent / "data" / "DSQA-full.csv"

        if not csv_path.exists():
            st.error(f"Dataset file not found: {csv_path}")
            st.info("Please ensure DSQA-full.csv is in the data/ directory")
            return []

        df = pd.read_csv(csv_path)

        # Filter out rows with missing answers
        df = df.dropna(subset=["answer"])

        # Filter for History category
        history_df = df[df["problem_category"] == "History"]

        # Pick 5 random questions
        if len(history_df) > 5:
            history_df = history_df.sample(n=5, random_state=random.randint(0, 10000))

        # Format for display
        history_questions = []
        icons = ["🏛️", "⚔️", "🗽", "👑", "🌍"]  # Predefined icons for variety

        for idx, (_, row) in enumerate(history_df.iterrows()):
            history_questions.append({
                "icon": icons[idx % len(icons)],
                "title": f"History Question {idx + 1}",
                "question": row["problem"],
                "example_id": row["example_id"],
                "answer": str(row["answer"])
            })

        return history_questions
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

# Load questions with a spinner if dataset is available
if DATASET_AVAILABLE:
    with st.spinner("Loading questions from DeepSearchQA dataset..."):
        history_questions = load_history_questions()

    if not history_questions:
        st.warning("No questions loaded. Please check the dataset configuration.")
        st.stop()
else:
    st.stop()

st.subheader("📚 Sample History Questions")
st.write("Click on a question to select it for processing")

# Initialize session state for selected question
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Create tiles in rows
col1, col2 = st.columns(2)

for idx, q in enumerate(history_questions):
    with col1 if idx % 2 == 0 else col2:
        # Create a clickable button styled as a tile
        button_label = f"{q['icon']} {q['title']}\n\n{q['question'][:100]}..." if len(q['question']) > 100 else f"{q['icon']} {q['title']}\n\n{q['question']}"

        if st.button(
                button_label,
                key=f"tile_{idx}",
                use_container_width=True,
                type="secondary"
        ):
            # Store selected question in session state
            st.session_state.selected_question = q
            st.rerun()

# Display selected question for processing
if st.session_state.selected_question:
    st.divider()
    st.subheader("🎯 Selected Question for Processing")

    selected = st.session_state.selected_question

    st.markdown(f"""
    <div style="padding: 25px; border-radius: 10px; border: 2px solid #4CAF50; background-color: #e8f5e9; margin-bottom: 20px;">
        <h3>{selected['icon']} {selected['title']}</h3>
        <p style="color: #333; font-size: 18px; margin-top: 15px;"><strong>Question:</strong></p>
        <p style="color: #555; font-size: 16px;">{selected['question']}</p>
        <p style="color: #666; font-size: 14px; margin-top: 15px;">
            <strong>Example ID:</strong> {selected['example_id']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize agent response in session state
    if 'agent_response' not in st.session_state:
        st.session_state.agent_response = None

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.button("🔄 Process with Agent", type="primary", use_container_width=True):
            if AGENT_AVAILABLE:
                with st.spinner("🤖 Agent is processing your question... This may take a minute."):
                    try:
                        # Create the agent
                        agent = KnowledgeGroundedAgent(enable_planning=True)

                        # Process the question
                        response = agent.answer(selected['question'])

                        # Store the response in session state
                        st.session_state.agent_response = response
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.error("Agent not available. Please install the required dependencies.")
    with col_b:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_question = None
            st.session_state.agent_response = None
            st.session_state.feedback_submitted = False
            st.rerun()

    # Display agent response if available
    if st.session_state.agent_response:
        st.divider()
        st.subheader("🤖 Agent Response")

        response = st.session_state.agent_response

        # Display the answer
        st.markdown("### 📝 Answer")
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: #f9f9f9; margin-bottom: 20px;">
            {response.text}
        </div>
        """, unsafe_allow_html=True)

        # Display execution details in expanders
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Duration", f"{response.total_duration_ms / 1000:.2f}s")
        with col2:
            st.metric("🔍 Sources Found", len(response.sources))
        with col3:
            st.metric("🔎 Search Queries", len(response.search_queries))

        # Search queries
        if response.search_queries:
            with st.expander("🔎 Search Queries Executed", expanded=False):
                for idx, query in enumerate(response.search_queries, 1):
                    st.markdown(f"{idx}. `{query}`")

        # Sources used
        if response.sources:
            with st.expander("📚 Sources Used", expanded=False):
                for idx, source in enumerate(response.sources, 1):
                    st.markdown(f"{idx}. [{source.title}]({source.url})")
                    if source.snippet:
                        st.caption(source.snippet[:200] + "...")

        # Research plan
        if response.plan and response.plan.steps:
            with st.expander("📋 Research Plan", expanded=False):
                st.markdown(f"**Reasoning:** {response.plan.reasoning}")
                st.markdown("**Steps:**")
                for step in response.plan.steps:
                    status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "skipped": "⏭️"}.get(step.status, "")
                    st.markdown(f"{status_icon} **Step {step.step_id}:** {step.description} (Tool: {step.suggested_tool})")

        # Ground truth comparison
        st.divider()
        st.markdown("### 📊 Ground Truth Comparison")
        col_gt, col_ag = st.columns(2)
        with col_gt:
            st.markdown("**Ground Truth Answer:**")
            st.info(selected['answer'])
        with col_ag:
            st.markdown("**Agent Answer:**")
            st.success(response.text[:500] + "..." if len(response.text) > 500 else response.text)

# Feedback Section - Only show if agent has responded
if st.session_state.get('agent_response') and st.session_state.get('selected_question'):
    st.divider()
    st.subheader("💬 Rate the Agent's Response")

    # Initialize feedback state
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False

    if not st.session_state.feedback_submitted:
        with st.form("feedback_form"):
            st.write("Was the agent's answer helpful and accurate?")

            # Thumbs up/down
            feedback = st.radio(
                "Your rating:",
                ["👍 Thumbs Up", "👎 Thumbs Down"],
                horizontal=True,
                label_visibility="collapsed"
            )

            # Submit button
            submitted = st.form_submit_button("Submit Feedback", type="primary")

            if submitted:
                # Store feedback value
                feedback_score = 1 if "Up" in feedback else 0
                st.session_state.feedback_submitted = True
                st.session_state.feedback_score = feedback_score

                # TODO: Send feedback to Langfuse
                # langfuse_client = AsyncClientManager.get_instance().langfuse_client
                # langfuse_client.create_score(
                #     value=feedback_score,
                #     name="User Feedback",
                #     comment=f"The user gave this response a thumbs {'up' if feedback_score else 'down'}.",
                #     trace_id=trace_id,
                # )

                st.success("✅ Thank you for your feedback!")
                st.balloons()
                st.info(f"You rated: {feedback}")
    else:
        st.success("✅ Thank you for your feedback!")
