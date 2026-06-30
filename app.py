import pandas as pd
import plotly.express as px
import streamlit as st

from utils.agents import message_agent, planner_agent, priority_agent, task_agent
from utils.document_reader import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks,
)
from utils.rag_engine import (
    calculate_rag_confidence,
    combine_retrieved_chunks,
    retrieve_relevant_chunks,
)


TASK_COLUMNS = ["Task", "Priority", "Status", "Deadline"]


st.set_page_config(
    page_title="Agentic AI Workflow Automation Platform",
    page_icon=":material/automation:",
    layout="wide",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070b18;
    --panel: rgba(15, 23, 42, 0.92);
    --panel-soft: rgba(30, 41, 59, 0.72);
    --border: rgba(148, 163, 184, 0.20);
    --text: #e5e7eb;
    --muted: #94a3b8;
    --cyan: #22d3ee;
    --blue: #3b82f6;
    --violet: #8b5cf6;
    --green: #22c55e;
    --orange: #f97316;
    --red: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(34, 211, 238, 0.16), transparent 28%),
        radial-gradient(circle at 85% 0%, rgba(139, 92, 246, 0.20), transparent 30%),
        radial-gradient(circle at 65% 85%, rgba(59, 130, 246, 0.12), transparent 32%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #111827 100%);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.98));
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.command-header {
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88)),
        radial-gradient(circle at 85% 20%, rgba(34, 211, 238, 0.20), transparent 30%);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 32px;
    margin-bottom: 22px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.35);
}

.command-kicker {
    color: var(--cyan);
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 12px;
    margin-bottom: 12px;
}

.command-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #f8fafc;
    font-size: 48px;
    line-height: 1.02;
    font-weight: 700;
    margin-bottom: 12px;
}

.command-subtitle {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
    max-width: 920px;
}

.status-strip {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 22px;
}

.status-pill {
    border: 1px solid rgba(34, 211, 238, 0.28);
    background: rgba(8, 47, 73, 0.38);
    color: #cffafe;
    padding: 9px 13px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
}

.metric-card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.76));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    min-height: 128px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.25);
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    color: #f8fafc;
    font-size: 36px;
    font-weight: 700;
    margin-top: 10px;
}

.metric-note {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 4px;
}

.panel {
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 60px rgba(0,0,0,0.28);
    margin-bottom: 20px;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #f8fafc;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 6px;
}

.panel-copy {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 18px;
}

.agent-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
}

.agent-card {
    background: rgba(2, 6, 23, 0.60);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 16px;
}

.agent-number {
    width: 34px;
    height: 34px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    color: #020617;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}

.agent-name {
    color: #f8fafc;
    font-weight: 800;
    margin-bottom: 6px;
}

.agent-status {
    color: #22c55e;
    font-size: 13px;
    font-weight: 800;
}

.agent-waiting {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 800;
}

.summary-box {
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 18px;
}

.label-small {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.value-line {
    color: #f8fafc;
    font-weight: 800;
    margin-bottom: 14px;
}

.evidence-box {
    background: rgba(8, 47, 73, 0.28);
    border-left: 4px solid var(--cyan);
    padding: 14px;
    border-radius: 14px;
    color: #dbeafe;
}

.report-box {
    background: rgba(2, 6, 23, 0.44);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 18px;
}

.small-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #06b6d4, #6366f1);
    color: white;
    font-weight: 900;
    padding: 13px;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 900;
    padding: 13px;
}

div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
    border-radius: 16px;
    overflow: hidden;
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
}

@media (max-width: 900px) {
    .command-title { font-size: 34px; }
    .agent-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    defaults = {
        "workflow_plan": "",
        "task_text": "",
        "priority_report": "",
        "message_report": "",
        "task_df": pd.DataFrame(columns=TASK_COLUMNS),
        "goal_history": [],
        "document_text": "",
        "document_chunks": [],
        "retrieved_chunks": [],
        "rag_context": "",
        "rag_confidence": 0,
        "current_goal": "",
        "current_role": "Student",
        "current_timeline": "7 Days",
        "company_name": "",
        "contact_person": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_task_text(task_text):
    tasks = []

    for line in task_text.split("\n"):
        clean_line = line.strip()

        if not clean_line or "|" not in clean_line:
            continue

        clean_line = clean_line.strip("|").strip()
        lower_line = clean_line.lower()

        if "task" in lower_line and "priority" in lower_line and "status" in lower_line:
            continue

        if set(clean_line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue

        parts = [part.strip() for part in clean_line.split("|") if part.strip()]

        if len(parts) >= 4:
            priority = parts[1].title()
            status = parts[2].title()

            if priority not in ["High", "Medium", "Low"]:
                priority = "Medium"

            if status not in ["Not Started", "In Progress", "Completed", "Blocked"]:
                status = "Not Started"

            tasks.append(
                {
                    "Task": parts[0],
                    "Priority": priority,
                    "Status": status,
                    "Deadline": parts[3],
                }
            )

    if not tasks:
        return pd.DataFrame(columns=TASK_COLUMNS)

    return pd.DataFrame(tasks)


def get_task_metrics(task_df):
    total_tasks = len(task_df)

    if total_tasks == 0:
        return 0, 0, 0, 0, 0, 0

    high_priority = len(task_df[task_df["Priority"] == "High"])
    completed = len(task_df[task_df["Status"] == "Completed"])
    in_progress = len(task_df[task_df["Status"] == "In Progress"])
    blocked = len(task_df[task_df["Status"] == "Blocked"])
    completion_percent = round((completed / total_tasks) * 100)

    return total_tasks, high_priority, completed, in_progress, blocked, completion_percent


def task_df_to_text(task_df):
    if task_df.empty:
        return "No tasks available."

    return task_df.to_string(index=False)


def create_status_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({"Status": ["No Tasks"], "Count": [0]})
    else:
        data = task_df["Status"].value_counts().reset_index()
        data.columns = ["Status", "Count"]

    fig = px.bar(
        data,
        x="Status",
        y="Count",
        title="Task Execution Status",
        text="Count",
    )

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


def create_priority_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({"Priority": ["No Tasks"], "Count": [0]})
    else:
        data = task_df["Priority"].value_counts().reset_index()
        data.columns = ["Priority", "Count"]

    fig = px.pie(
        data,
        names="Priority",
        values="Count",
        title="Priority Distribution",
        hole=0.55,
    )

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


def create_deadline_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({"Deadline": ["No Tasks"], "Count": [0]})
    else:
        data = task_df["Deadline"].value_counts().reset_index()
        data.columns = ["Deadline", "Count"]

    fig = px.bar(
        data,
        x="Deadline",
        y="Count",
        title="Deadline Load",
        text="Count",
    )

    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    return fig


def get_agent_status():
    return {
        "RAG Engine": bool(st.session_state.rag_context),
        "Planner": bool(st.session_state.workflow_plan),
        "Task": not st.session_state.task_df.empty,
        "Priority": bool(st.session_state.priority_report),
        "Message": bool(st.session_state.message_report),
    }


def build_final_report():
    total_tasks, high_priority, completed, in_progress, blocked, completion_percent = (
        get_task_metrics(st.session_state.task_df)
    )

    report = f"""
# Agentic AI Workflow Automation Report

## Goal
{st.session_state.current_goal}

## Role
{st.session_state.current_role}

## Timeline
{st.session_state.current_timeline}

## Target Company
{st.session_state.company_name}

## Contact Person
{st.session_state.contact_person}

## Automation Metrics
Total Tasks: {total_tasks}
High Priority Tasks: {high_priority}
In Progress Tasks: {in_progress}
Completed Tasks: {completed}
Blocked Tasks: {blocked}
Completion Percentage: {completion_percent}%

## RAG Confidence
{st.session_state.rag_confidence}

## Workflow Plan
{st.session_state.workflow_plan}

## Task Checklist
{task_df_to_text(st.session_state.task_df)}

## Priority Agent Report
{st.session_state.priority_report}

## Message Agent Drafts
{st.session_state.message_report}

## Final Recommendation
Start with high-priority tasks, complete blocked items early, review outreach messages, and follow the timeline step by step.
"""

    return report


def clear_workspace():
    st.session_state.workflow_plan = ""
    st.session_state.task_text = ""
    st.session_state.priority_report = ""
    st.session_state.message_report = ""
    st.session_state.task_df = pd.DataFrame(columns=TASK_COLUMNS)
    st.session_state.document_text = ""
    st.session_state.document_chunks = []
    st.session_state.retrieved_chunks = []
    st.session_state.rag_context = ""
    st.session_state.rag_confidence = 0
    st.session_state.current_goal = ""
    st.session_state.company_name = ""
    st.session_state.contact_person = ""


def render_metric(label, value, note):
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_agent_pipeline(agent_status):
    cards = [
        ("01", "RAG Engine", "Document context retrieval"),
        ("02", "Planner", "Goal to workflow plan"),
        ("03", "Task", "Checklist generation"),
        ("04", "Priority", "Execution guidance"),
        ("05", "Message", "Outreach drafts"),
    ]

    html = '<div class="agent-grid">'

    for number, name, desc in cards:
        done = agent_status.get(name, False)
        status_text = "Completed" if done else "Waiting"
        status_class = "agent-status" if done else "agent-waiting"

        html += f"""
        <div class="agent-card">
            <div class="agent-number">{number}</div>
            <div class="agent-name">{name}</div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.5;">{desc}</div>
            <div class="{status_class}">{status_text}</div>
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


init_session_state()

total_tasks, high_priority, completed, in_progress, blocked, completion_percent = (
    get_task_metrics(st.session_state.task_df)
)
agent_status = get_agent_status()
completed_agents = sum(1 for status in agent_status.values() if status)

with st.sidebar:
    st.markdown("## Command Controls")
    st.caption("Enter a goal and run the multi-agent automation workflow.")

    goal = st.text_area(
        "Goal",
        placeholder="Example: Apply for Google AI Internship in 7 days",
        height=120,
    )

    role = st.selectbox(
        "Role",
        [
            "Student",
            "Fresher",
            "AI/ML Intern",
            "Software Engineer Intern",
            "Data Analyst",
            "Content Creator",
            "Startup Founder",
        ],
    )

    timeline = st.selectbox(
        "Timeline",
        ["1 Day", "3 Days", "7 Days", "2 Weeks", "1 Month"],
        index=2,
    )

    company_name = st.text_input(
        "Target Company",
        placeholder="Example: Google",
    )

    contact_person = st.text_input(
        "Recruiter / Contact",
        placeholder="Example: Hiring Manager",
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
    )

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            st.session_state.document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.document_text = extract_text_from_txt(uploaded_file)

        st.session_state.document_chunks = split_text_into_chunks(
            st.session_state.document_text
        )

        st.success("Document extracted.")
        st.caption(f"Words: {len(st.session_state.document_text.split())}")
        st.caption(f"Chunks: {len(st.session_state.document_chunks)}")

    top_k = st.slider("Evidence sections", 1, 5, 3)

    run_button = st.button("Run Automation Workflow")
    clear_button = st.button("Clear Workspace")

    if clear_button:
        clear_workspace()
        st.success("Workspace cleared.")


st.markdown(
    """
<div class="command-header">
    <div class="command-kicker">Week 4 Day 6 - Professional SaaS Redesign</div>
    <div class="command-title">Agentic Workflow Command Center</div>
    <div class="command-subtitle">
        A multi-agent AI platform that converts goals into structured plans, task boards,
        priority insights, outreach messages, and downloadable automation reports using RAG-powered context.
    </div>
    <div class="status-strip">
        <div class="status-pill">Planner Agent</div>
        <div class="status-pill">Task Agent</div>
        <div class="status-pill">Priority Agent</div>
        <div class="status-pill">Message Agent</div>
        <div class="status-pill">RAG Engine</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    render_metric("Total Tasks", total_tasks, "Generated checklist")

with m2:
    render_metric("Progress", f"{completion_percent}%", "Completed tasks")

with m3:
    render_metric("High Priority", high_priority, "Needs attention")

with m4:
    render_metric("Blocked", blocked, "Execution risks")

with m5:
    render_metric("RAG Score", st.session_state.rag_confidence, "Context match")

with m6:
    render_metric("Agents", f"{completed_agents}/5", "Workflow status")

st.progress(completion_percent / 100)
st.markdown("---")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown(
        """
    <div class="panel">
        <div class="panel-title">Agent Pipeline</div>
        <div class="panel-copy">
            This pipeline shows how each AI agent contributes to the automation workflow.
        </div>
    """,
        unsafe_allow_html=True,
    )
    render_agent_pipeline(agent_status)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        f"""
    <div class="panel">
        <div class="panel-title">Automation Summary</div>
        <div class="panel-copy">
            Current goal and execution status.
        </div>
        <div class="small-label">Current Goal</div>
        <div style="color:#f8fafc; font-weight:800; margin-bottom:12px;">
            {st.session_state.current_goal if st.session_state.current_goal else "No goal generated yet"}
        </div>
        <div class="small-label">Execution Health</div>
        <div style="color:#22c55e; font-weight:900; font-size:22px;">
            {"Active Workflow" if st.session_state.workflow_plan else "Waiting"}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


if run_button:
    if not goal.strip():
        st.error("Please enter a goal first.")
    else:
        st.session_state.current_goal = goal
        st.session_state.current_role = role
        st.session_state.current_timeline = timeline
        st.session_state.company_name = company_name
        st.session_state.contact_person = contact_person

        if st.session_state.document_chunks:
            with st.spinner("RAG Engine is retrieving document context..."):
                st.session_state.retrieved_chunks = retrieve_relevant_chunks(
                    goal,
                    st.session_state.document_chunks,
                    top_k=top_k,
                )

                st.session_state.rag_context = combine_retrieved_chunks(
                    st.session_state.retrieved_chunks
                )

                st.session_state.rag_confidence = calculate_rag_confidence(
                    st.session_state.retrieved_chunks
                )
        else:
            st.session_state.retrieved_chunks = []
            st.session_state.rag_context = "No document uploaded. Agents used goal-only context."
            st.session_state.rag_confidence = 0

        with st.spinner("Planner Agent is creating workflow plan..."):
            st.session_state.workflow_plan = planner_agent(
                goal,
                role,
                timeline,
                st.session_state.rag_context,
            )

        with st.spinner("Task Agent is creating task board..."):
            st.session_state.task_text = task_agent(
                goal,
                role,
                timeline,
                st.session_state.workflow_plan,
                st.session_state.rag_context,
            )

        st.session_state.task_df = parse_task_text(st.session_state.task_text)

        with st.spinner("Priority Agent is analyzing execution risks..."):
            st.session_state.priority_report = priority_agent(
                goal,
                role,
                timeline,
                st.session_state.workflow_plan,
                task_df_to_text(st.session_state.task_df),
                st.session_state.rag_context,
            )

        with st.spinner("Message Agent is drafting outreach messages..."):
            st.session_state.message_report = message_agent(
                goal,
                role,
                timeline,
                company_name,
                contact_person,
                st.session_state.workflow_plan,
                task_df_to_text(st.session_state.task_df),
                st.session_state.priority_report,
                st.session_state.rag_context,
            )

        st.session_state.goal_history.insert(0, goal)
        st.session_state.goal_history = st.session_state.goal_history[:5]

        st.success("Automation workflow completed successfully.")
        st.rerun()


st.markdown("---")

chart1, chart2, chart3 = st.columns(3, gap="large")

with chart1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(create_status_chart(st.session_state.task_df), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with chart2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(create_priority_chart(st.session_state.task_df), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with chart3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(create_deadline_chart(st.session_state.task_df), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Workflow Plan",
        "Task Board",
        "Priority Insights",
        "Message Drafts",
        "RAG Evidence",
        "Goal History",
        "Final Report",
    ]
)

with tab1:
    st.markdown("## Workflow Plan")

    if st.session_state.workflow_plan:
        st.markdown(st.session_state.workflow_plan)
        st.download_button(
            label="Download Workflow Plan",
            data=st.session_state.workflow_plan,
            file_name="workflow_plan.txt",
            mime="text/plain",
        )
    else:
        st.info("Workflow plan will appear after running the automation workflow.")

with tab2:
    st.markdown("## Editable Task Board")

    if not st.session_state.task_df.empty:
        edited_df = st.data_editor(
            st.session_state.task_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Not Started", "In Progress", "Completed", "Blocked"],
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["High", "Medium", "Low"],
                ),
            },
        )

        st.session_state.task_df = edited_df
        csv_data = edited_df.to_csv(index=False).encode("utf-8")

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                label="Download Task Board CSV",
                data=csv_data,
                file_name="workflow_task_board.csv",
                mime="text/csv",
            )

        with c2:
            if st.button("Re-analyze Updated Tasks"):
                with st.spinner("Priority Agent is re-analyzing updated tasks..."):
                    st.session_state.priority_report = priority_agent(
                        st.session_state.current_goal,
                        st.session_state.current_role,
                        st.session_state.current_timeline,
                        st.session_state.workflow_plan,
                        task_df_to_text(st.session_state.task_df),
                        st.session_state.rag_context,
                    )

                st.success("Priority insights updated.")
                st.rerun()
    else:
        st.info("Task board will appear after generation.")

with tab3:
    st.markdown("## Priority Insights")

    if st.session_state.priority_report:
        st.markdown(st.session_state.priority_report)
        st.download_button(
            label="Download Priority Insights",
            data=st.session_state.priority_report,
            file_name="priority_insights.txt",
            mime="text/plain",
        )
    else:
        st.info("Priority insights will appear after generation.")

with tab4:
    st.markdown("## Message Drafts")

    if st.session_state.message_report:
        st.markdown(st.session_state.message_report)
        st.download_button(
            label="Download Message Drafts",
            data=st.session_state.message_report,
            file_name="message_drafts.txt",
            mime="text/plain",
        )
    else:
        st.info("Message drafts will appear after generation.")

with tab5:
    st.markdown("## RAG Evidence")
    st.markdown(
        f"""
        <div class="evidence-box">
            <b>RAG Confidence Score:</b> {st.session_state.rag_confidence}<br>
            <b>Evidence Sections:</b> {len(st.session_state.retrieved_chunks)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.retrieved_chunks:
        for item in st.session_state.retrieved_chunks:
            with st.expander(f"{item['source']} | Similarity Score: {item['score']}"):
                st.write(item["chunk"])
    else:
        st.info("Relevant document evidence will appear after running with a document.")

with tab6:
    st.markdown("## Goal History")

    if st.session_state.goal_history:
        for index, item in enumerate(st.session_state.goal_history, start=1):
            st.write(f"{index}. {item}")
    else:
        st.info("Goal history will appear after generation.")

with tab7:
    st.markdown("## Final Automation Report")

    if st.session_state.workflow_plan:
        final_report = build_final_report()
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(final_report)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            label="Download Complete Automation Report",
            data=final_report,
            file_name="agentic_ai_workflow_automation_report.txt",
            mime="text/plain",
        )
    else:
        st.info("Final report will appear after running the workflow.")
