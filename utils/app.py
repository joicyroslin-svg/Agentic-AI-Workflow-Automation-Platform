import streamlit as st
import pandas as pd
import plotly.express as px

from utils.agents import planner_agent, task_agent
from utils.document_reader import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks
)
from utils.rag_engine import (
    retrieve_relevant_chunks,
    combine_retrieved_chunks,
    calculate_rag_confidence
)


st.set_page_config(
    page_title="Agentic AI Workflow Automation Platform",
    page_icon="⚙️",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(37, 99, 235, 0.12), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(124, 58, 237, 0.12), transparent 28%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #fdf2f8 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    background:
        radial-gradient(circle at 85% 10%, rgba(249, 115, 22, 0.25), transparent 25%),
        linear-gradient(135deg, #0f172a, #312e81 55%, #4c1d95);
    color: white;
    padding: 34px;
    border-radius: 28px;
    box-shadow: 0 28px 70px rgba(15, 23, 42, 0.25);
    margin-bottom: 24px;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 14px;
}

.title {
    font-size: 46px;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 12px;
}

.subtitle {
    font-size: 17px;
    color: #dbeafe;
    line-height: 1.6;
    max-width: 900px;
}

.card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-value {
    color: #0f172a;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}

.section-copy {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 16px;
}

.rag-box {
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 800;
    padding: 13px;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 800;
    padding: 13px;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "workflow_plan" not in st.session_state:
        st.session_state.workflow_plan = ""

    if "task_text" not in st.session_state:
        st.session_state.task_text = ""

    if "task_df" not in st.session_state:
        st.session_state.task_df = pd.DataFrame(
            columns=["Task", "Priority", "Status", "Deadline"]
        )

    if "goal_history" not in st.session_state:
        st.session_state.goal_history = []

    if "document_text" not in st.session_state:
        st.session_state.document_text = ""

    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []

    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []

    if "rag_context" not in st.session_state:
        st.session_state.rag_context = ""

    if "rag_confidence" not in st.session_state:
        st.session_state.rag_confidence = 0


def parse_task_text(task_text):
    tasks = []

    for line in task_text.split("\n"):
        clean_line = line.strip().strip("|").strip()

        if not clean_line:
            continue

        if "|" not in clean_line:
            continue

        lower_line = clean_line.lower()

        if "task" in lower_line and "priority" in lower_line and "status" in lower_line:
            continue

        if set(clean_line.replace("|", "").replace("-", "").strip()) == set():
            continue

        parts = [part.strip() for part in clean_line.split("|")]

        if len(parts) >= 4:
            task = {
                "Task": parts[0],
                "Priority": parts[1],
                "Status": parts[2],
                "Deadline": parts[3]
            }
            tasks.append(task)

    if not tasks:
        return pd.DataFrame(columns=["Task", "Priority", "Status", "Deadline"])

    return pd.DataFrame(tasks)


def get_task_metrics(task_df):
    total_tasks = len(task_df)

    if total_tasks == 0:
        return 0, 0, 0, 0

    high_priority = len(task_df[task_df["Priority"] == "High"])
    completed = len(task_df[task_df["Status"] == "Completed"])
    in_progress = len(task_df[task_df["Status"] == "In Progress"])

    return total_tasks, high_priority, completed, in_progress


def create_status_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({
            "Status": ["No Tasks"],
            "Count": [0]
        })
    else:
        data = task_df["Status"].value_counts().reset_index()
        data.columns = ["Status", "Count"]

    fig = px.bar(
        data,
        x="Status",
        y="Count",
        title="Task Status Overview",
        text="Count"
    )

    fig.update_layout(height=350)

    return fig


def create_priority_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({
            "Priority": ["No Tasks"],
            "Count": [0]
        })
    else:
        data = task_df["Priority"].value_counts().reset_index()
        data.columns = ["Priority", "Count"]

    fig = px.pie(
        data,
        names="Priority",
        values="Count",
        title="Task Priority Distribution"
    )

    fig.update_layout(height=350)

    return fig


def clear_workspace():
    st.session_state.workflow_plan = ""
    st.session_state.task_text = ""
    st.session_state.task_df = pd.DataFrame(
        columns=["Task", "Priority", "Status", "Deadline"]
    )
    st.session_state.document_text = ""
    st.session_state.document_chunks = []
    st.session_state.retrieved_chunks = []
    st.session_state.rag_context = ""
    st.session_state.rag_confidence = 0


init_session_state()

total_tasks, high_priority, completed, in_progress = get_task_metrics(
    st.session_state.task_df
)


st.markdown("""
<div class="hero">
    <div class="badge">Week 4 Day 3 — RAG Context Upgrade</div>
    <div class="title">Agentic AI Workflow Automation Platform</div>
    <div class="subtitle">
        Convert goals into workflow plans, retrieve useful document context,
        generate AI task checklists, and track progress from one dashboard.
    </div>
</div>
""", unsafe_allow_html=True)


m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Tasks</div>
        <div class="metric-value">{total_tasks}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Priority</div>
        <div class="metric-value">{high_priority}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">In Progress</div>
        <div class="metric-value">{in_progress}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Completed</div>
        <div class="metric-value">{completed}</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RAG Score</div>
        <div class="metric-value">{st.session_state.rag_confidence}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Goal + Document Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Enter a goal and optionally upload a PDF/TXT document for RAG-based planning.</div>',
        unsafe_allow_html=True
    )

    goal = st.text_area(
        "Enter your goal",
        placeholder="Example: Prepare and apply for Google AI Internship in 7 days",
        height=130
    )

    role = st.selectbox(
        "Select your role",
        [
            "Student",
            "Fresher",
            "AI/ML Intern",
            "Software Engineer Intern",
            "Data Analyst",
            "Content Creator",
            "Startup Founder"
        ]
    )

    timeline = st.selectbox(
        "Timeline",
        [
            "1 Day",
            "3 Days",
            "7 Days",
            "2 Weeks",
            "1 Month"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT document",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.document_text = extract_text_from_txt(uploaded_file)

        st.session_state.document_chunks = split_text_into_chunks(
            st.session_state.document_text
        )

        st.success("Document uploaded and extracted successfully.")
        st.caption(f"Extracted words: {len(st.session_state.document_text.split())}")
        st.caption(f"Document chunks created: {len(st.session_state.document_chunks)}")

    top_k = st.slider(
        "Number of document sections to use",
        1,
        5,
        3
    )

    run_button = st.button("Generate RAG Workflow + Tasks")
    clear_button = st.button("Clear Workspace")

    if clear_button:
        clear_workspace()
        st.success("Workspace cleared.")

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Day 3 Features</div>', unsafe_allow_html=True)

    st.write("""
    Day 3 adds RAG context retrieval:

    - PDF/TXT upload
    - Document text extraction
    - Text chunking
    - Relevant chunk retrieval
    - RAG confidence score
    - Planner Agent uses document context
    - Task Agent uses document context
    """)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            create_status_chart(st.session_state.task_df),
            use_container_width=True
        )

    with chart_col2:
        st.plotly_chart(
            create_priority_chart(st.session_state.task_df),
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


if run_button:
    if not goal.strip():
        st.error("Please enter a goal first.")

    else:
        if st.session_state.document_chunks:
            with st.spinner("Finding relevant document context using RAG..."):
                st.session_state.retrieved_chunks = retrieve_relevant_chunks(
                    goal,
                    st.session_state.document_chunks,
                    top_k=top_k
                )

                st.session_state.rag_context = combine_retrieved_chunks(
                    st.session_state.retrieved_chunks
                )

                st.session_state.rag_confidence = calculate_rag_confidence(
                    st.session_state.retrieved_chunks
                )
        else:
            st.session_state.retrieved_chunks = []
            st.session_state.rag_context = "No document uploaded. Planner and Task Agent used goal-only context."
            st.session_state.rag_confidence = 0

        with st.spinner("Planner Agent is creating workflow plan..."):
            st.session_state.workflow_plan = planner_agent(
                goal,
                role,
                timeline,
                st.session_state.rag_context
            )

        with st.spinner("Task Agent is creating task checklist..."):
            st.session_state.task_text = task_agent(
                goal,
                role,
                timeline,
                st.session_state.workflow_plan,
                st.session_state.rag_context
            )

        st.session_state.task_df = parse_task_text(st.session_state.task_text)

        st.session_state.goal_history.insert(0, goal)
        st.session_state.goal_history = st.session_state.goal_history[:5]

        st.success("RAG workflow plan and task checklist generated successfully.")
        st.rerun()


st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Workflow Plan",
    "Task Checklist",
    "RAG Context",
    "Goal History",
    "Project Notes"
])

with tab1:
    st.markdown("## AI Workflow Plan")

    if st.session_state.workflow_plan:
        st.markdown(st.session_state.workflow_plan)

        st.download_button(
            label="Download Workflow Plan",
            data=st.session_state.workflow_plan,
            file_name="workflow_plan.txt",
            mime="text/plain"
        )

    else:
        st.info("Workflow plan will appear here.")


with tab2:
    st.markdown("## Editable Task Checklist")

    if not st.session_state.task_df.empty:
        edited_df = st.data_editor(
            st.session_state.task_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Not Started", "In Progress", "Completed", "Blocked"]
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["High", "Medium", "Low"]
                )
            }
        )

        st.session_state.task_df = edited_df

        csv_data = edited_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Task Checklist CSV",
            data=csv_data,
            file_name="workflow_tasks.csv",
            mime="text/csv"
        )

    else:
        st.info("Task checklist will appear here after generation.")


with tab3:
    st.markdown("## Retrieved RAG Context")

    st.write(f"RAG Confidence Score: {st.session_state.rag_confidence}")

    if st.session_state.retrieved_chunks:
        for item in st.session_state.retrieved_chunks:
            with st.expander(f"{item['source']} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])
    else:
        st.info("Relevant document sections will appear here after running the workflow.")


with tab4:
    st.markdown("## Goal History")

    if st.session_state.goal_history:
        for index, item in enumerate(st.session_state.goal_history, start=1):
            st.write(f"{index}. {item}")
    else:
        st.info("Goal history will appear here.")


with tab5:
    st.markdown("## Week 4 Day 3 Notes")

    st.write("""
    Day 3 added RAG context retrieval.

    The app can now upload PDF/TXT files, extract text, split the document into chunks,
    find relevant chunks using similarity search, and pass that retrieved context to the Planner Agent and Task Agent.

    This makes the workflow more personalized and document-grounded.
    """)