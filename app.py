from html import escape

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
STATUS_OPTIONS = ["Not Started", "In Progress", "Completed", "Blocked"]
PRIORITY_OPTIONS = ["High", "Medium", "Low"]


st.set_page_config(
    page_title="Agentic AI Workflow Automation Platform",
    page_icon="⚙️",
    layout="wide",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: rgba(148, 163, 184, 0.22);
    --panel: rgba(255, 255, 255, 0.78);
    --blue: #2563eb;
    --cyan: #06b6d4;
    --green: #16a34a;
    --amber: #f59e0b;
    --red: #ef4444;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 8% 4%, rgba(37, 99, 235, 0.16), transparent 28%),
        radial-gradient(circle at 88% 8%, rgba(6, 182, 212, 0.15), transparent 30%),
        radial-gradient(circle at 50% 92%, rgba(99, 102, 241, 0.12), transparent 34%),
        linear-gradient(135deg, #f8fbff 0%, #eef5ff 46%, #fffafe 100%);
}

.block-container {
    max-width: 1540px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.88)),
        radial-gradient(circle at 20% 0%, rgba(37, 99, 235, 0.13), transparent 34%);
    border-right: 1px solid rgba(148, 163, 184, 0.24);
    box-shadow: 14px 0 44px rgba(15, 23, 42, 0.07);
}

section[data-testid="stSidebar"] label {
    color: #334155;
    font-weight: 800;
}

.sidebar-brand {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 24px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 18px 44px rgba(37, 99, 235, 0.12);
}

.brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 900;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    margin-bottom: 12px;
}

.brand-title {
    font-family: "Plus Jakarta Sans", sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #0f172a;
}

.brand-subtitle {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
    margin-top: 3px;
}

.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 34px;
    padding: 34px;
    border: 1px solid rgba(255, 255, 255, 0.86);
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(239, 246, 255, 0.78)),
        radial-gradient(circle at 88% 22%, rgba(37, 99, 235, 0.18), transparent 28%);
    box-shadow: 0 30px 80px rgba(37, 99, 235, 0.13);
    margin-bottom: 22px;
}

.kicker {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 8px 12px;
    color: #1d4ed8;
    background: rgba(37, 99, 235, 0.09);
    border: 1px solid rgba(37, 99, 235, 0.15);
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.hero-title {
    font-family: "Plus Jakarta Sans", sans-serif;
    font-size: 48px;
    line-height: 1.05;
    font-weight: 800;
    color: #0f172a;
    max-width: 960px;
}

.hero-subtitle {
    color: #475569;
    font-size: 17px;
    line-height: 1.7;
    margin-top: 12px;
    max-width: 860px;
}

.chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 24px;
}

.chip {
    border-radius: 999px;
    padding: 9px 13px;
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.20);
    color: #334155;
    font-size: 12px;
    font-weight: 800;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.metric-card, .mini-card, .report-shell {
    background: var(--panel);
    border: 1px solid rgba(255, 255, 255, 0.88);
    box-shadow: 0 22px 58px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(18px);
}

.metric-card {
    border-radius: 26px;
    padding: 20px;
    min-height: 142px;
}

.metric-icon {
    width: 40px;
    height: 40px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(6, 182, 212, 0.16));
    margin-bottom: 12px;
    font-size: 20px;
}

.metric-label, .eyebrow {
    color: #64748b;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    font-family: "Plus Jakarta Sans", sans-serif;
    color: #0f172a;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #64748b;
    font-size: 13px;
    margin-top: 4px;
}

.section-title {
    font-family: "Plus Jakarta Sans", sans-serif;
    color: #0f172a;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-copy {
    color: #64748b;
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 10px;
}

.mini-card {
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 16px;
}

.summary-value {
    color: #0f172a;
    font-size: 15px;
    font-weight: 800;
    line-height: 1.45;
    margin-top: 7px;
}

.evidence-box {
    border-radius: 18px;
    padding: 15px;
    background: rgba(236, 254, 255, 0.78);
    border: 1px solid rgba(6, 182, 212, 0.20);
    border-left: 5px solid #06b6d4;
    color: #164e63;
}

.action-box {
    border-radius: 20px;
    padding: 16px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(6, 182, 212, 0.12));
    border: 1px solid rgba(37, 99, 235, 0.15);
}

.report-shell {
    border-radius: 24px;
    padding: 22px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(148, 163, 184, 0.22);
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.58);
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.06);
}

.stButton > button {
    width: 100%;
    border-radius: 16px;
    border: 0;
    color: #ffffff;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    font-weight: 900;
    padding: 13px;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.22);
}

.stDownloadButton > button {
    width: 100%;
    border-radius: 16px;
    font-weight: 900;
    padding: 12px;
}

button[data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 14px;
    font-weight: 800;
}

div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
    border-radius: 18px;
    overflow: hidden;
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
}

@media (max-width: 1100px) {
    .hero-title { font-size: 34px; }
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
        "upload_key": 0,
        "goal_input": "",
        "role_input": "Student",
        "timeline_input": "7 Days",
        "company_input": "",
        "contact_input": "",
        "top_k_input": 3,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_task_text(task_text):
    tasks = []

    for line in task_text.splitlines():
        clean_line = line.strip().strip("|").strip()

        if not clean_line or "|" not in clean_line:
            continue

        normalized = clean_line.lower()

        if "task" in normalized and "priority" in normalized and "status" in normalized:
            continue

        if not clean_line.replace("|", "").replace("-", "").replace(":", "").strip():
            continue

        parts = [part.strip() for part in clean_line.split("|")]
        parts = [part for part in parts if part]

        if len(parts) < 4:
            continue

        priority = parts[1].title()
        status = parts[2].title()

        if priority not in PRIORITY_OPTIONS:
            priority = "Medium"

        if status not in STATUS_OPTIONS:
            status = "Not Started"

        tasks.append(
            {
                "Task": parts[0],
                "Priority": priority,
                "Status": status,
                "Deadline": parts[3],
            }
        )

    return pd.DataFrame(tasks, columns=TASK_COLUMNS)


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
    status_order = STATUS_OPTIONS
    if task_df.empty:
        data = pd.DataFrame({"Status": status_order, "Count": [0, 0, 0, 0]})
    else:
        data = task_df["Status"].value_counts().reindex(status_order, fill_value=0).reset_index()
        data.columns = ["Status", "Count"]

    fig = px.bar(
        data,
        x="Status",
        y="Count",
        text="Count",
        color="Status",
        color_discrete_map={
            "Not Started": "#94a3b8",
            "In Progress": "#2563eb",
            "Completed": "#16a34a",
            "Blocked": "#ef4444",
        },
    )

    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig.update_layout(
        height=320,
        title="Task Status",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter"),
        showlegend=False,
        margin=dict(l=14, r=14, t=54, b=14),
        title_font=dict(size=17, color="#0f172a", family="Plus Jakarta Sans"),
    )
    fig.update_xaxes(title=None, gridcolor="rgba(148,163,184,0.14)")
    fig.update_yaxes(title=None, gridcolor="rgba(148,163,184,0.22)")

    return fig


def create_priority_chart(task_df):
    if task_df.empty:
        data = pd.DataFrame({"Priority": PRIORITY_OPTIONS, "Count": [0, 0, 0]})
    else:
        data = task_df["Priority"].value_counts().reindex(PRIORITY_OPTIONS, fill_value=0).reset_index()
        data.columns = ["Priority", "Count"]

    fig = px.pie(
        data,
        names="Priority",
        values="Count",
        hole=0.58,
        color="Priority",
        color_discrete_map={
            "High": "#ef4444",
            "Medium": "#f59e0b",
            "Low": "#16a34a",
        },
    )

    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#ffffff", width=3)))
    fig.update_layout(
        height=320,
        title="Priority Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter"),
        margin=dict(l=14, r=14, t=54, b=14),
        title_font=dict(size=17, color="#0f172a", family="Plus Jakarta Sans"),
        legend=dict(orientation="h", y=-0.08),
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
        text="Count",
        color_discrete_sequence=["#2563eb"],
    )

    fig.update_traces(
        marker=dict(color="#2563eb", line_width=0),
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(
        height=320,
        title="Deadline Load",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", family="Inter"),
        showlegend=False,
        margin=dict(l=14, r=14, t=54, b=14),
        title_font=dict(size=17, color="#0f172a", family="Plus Jakarta Sans"),
    )
    fig.update_xaxes(title=None, gridcolor="rgba(148,163,184,0.14)")
    fig.update_yaxes(title=None, gridcolor="rgba(148,163,184,0.22)")

    return fig


def get_agent_status():
    return {
        "RAG Engine": bool(st.session_state.rag_context),
        "Planner": bool(st.session_state.workflow_plan),
        "Task": not st.session_state.task_df.empty,
        "Priority": bool(st.session_state.priority_report),
        "Message": bool(st.session_state.message_report),
        "Final Report": bool(st.session_state.workflow_plan),
    }


def build_final_report():
    total_tasks, high_priority, completed, in_progress, blocked, completion_percent = (
        get_task_metrics(st.session_state.task_df)
    )

    return f"""# Agentic AI Workflow Automation Report

## Goal
{st.session_state.current_goal}

## Role
{st.session_state.current_role}

## Timeline
{st.session_state.current_timeline}

## Target Company
{st.session_state.company_name or "Not provided"}

## Contact Person
{st.session_state.contact_person or "Not provided"}

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

## Task Board
{task_df_to_text(st.session_state.task_df)}

## Priority Report
{st.session_state.priority_report}

## Message Drafts
{st.session_state.message_report}

## Final Recommendation
Start with high-priority tasks first, remove blockers early, personalize outreach messages, and follow the selected timeline step by step.
"""


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
    st.session_state.current_role = "Student"
    st.session_state.current_timeline = "7 Days"
    st.session_state.company_name = ""
    st.session_state.contact_person = ""
    st.session_state.goal_input = ""
    st.session_state.role_input = "Student"
    st.session_state.timeline_input = "7 Days"
    st.session_state.company_input = ""
    st.session_state.contact_input = ""
    st.session_state.top_k_input = 3
    st.session_state.upload_key += 1


def render_metric_card(icon, label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{escape(str(icon))}</div>
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_timeline(agent_status):
    agents = [
        ("📄", "01", "RAG Engine", "Retrieves document context and evidence.", agent_status.get("RAG Engine", False)),
        ("🧭", "02", "Planner Agent", "Builds a practical workflow plan.", agent_status.get("Planner", False)),
        ("📋", "03", "Task Agent", "Creates the editable task board.", agent_status.get("Task", False)),
        ("🔥", "04", "Priority Agent", "Ranks urgency and execution risks.", agent_status.get("Priority", False)),
        ("💬", "05", "Message Agent", "Drafts outreach communication.", agent_status.get("Message", False)),
        ("📦", "06", "Final Report", "Packages the automation output.", agent_status.get("Final Report", False)),
    ]

    cols = st.columns(6)

    for col, (icon, number, title, description, is_done) in zip(cols, agents):
        status_text = "Completed" if is_done else "Waiting"
        status_emoji = "✅" if is_done else "⏳"

        with col:
            with st.container(border=True):
                st.markdown(f"### {icon}")
                st.markdown(f"**{number} · {title}**")
                st.caption(description)
                st.markdown(f"{status_emoji} **{status_text}**")


def render_section_header(title, copy):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(label, value):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="eyebrow">{escape(str(label))}</div>
            <div class="summary-value">{escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_next_best_action():
    if st.session_state.task_df.empty:
        return "Run the agent workflow to generate the plan, task board, and outreach drafts."

    high_priority_tasks = st.session_state.task_df[
        (st.session_state.task_df["Priority"] == "High")
        & (st.session_state.task_df["Status"] != "Completed")
    ]

    if not high_priority_tasks.empty:
        return str(high_priority_tasks.iloc[0]["Task"])

    incomplete_tasks = st.session_state.task_df[
        st.session_state.task_df["Status"] != "Completed"
    ]

    if not incomplete_tasks.empty:
        return str(incomplete_tasks.iloc[0]["Task"])

    return "All generated tasks are completed. Review the final report and send your outreach."


def run_agent_workflow(goal, role, timeline, company_name, contact_person, uploaded_file, top_k):
    st.session_state.current_goal = goal
    st.session_state.current_role = role
    st.session_state.current_timeline = timeline
    st.session_state.company_name = company_name
    st.session_state.contact_person = contact_person

    if uploaded_file:
        with st.spinner("Reading uploaded document..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                st.session_state.document_text = extract_text_from_pdf(uploaded_file)
            else:
                st.session_state.document_text = extract_text_from_txt(uploaded_file)

            st.session_state.document_chunks = split_text_into_chunks(
                st.session_state.document_text
            )

        if st.session_state.document_chunks:
            with st.spinner("RAG Engine is retrieving relevant document context..."):
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
            st.session_state.rag_context = "Uploaded document did not contain readable text."
            st.session_state.rag_confidence = 0
    else:
        st.session_state.document_text = ""
        st.session_state.document_chunks = []
        st.session_state.retrieved_chunks = []
        st.session_state.rag_context = "No document uploaded. Agents used goal-only context."
        st.session_state.rag_confidence = 0

    with st.spinner("Planner Agent is creating the workflow plan..."):
        st.session_state.workflow_plan = planner_agent(
            goal,
            role,
            timeline,
            st.session_state.rag_context,
        )

    with st.spinner("Task Agent is creating the editable task board..."):
        st.session_state.task_text = task_agent(
            goal,
            role,
            timeline,
            st.session_state.workflow_plan,
            st.session_state.rag_context,
        )

    st.session_state.task_df = parse_task_text(st.session_state.task_text)

    with st.spinner("Priority Agent is analyzing execution priorities..."):
        st.session_state.priority_report = priority_agent(
            goal,
            role,
            timeline,
            st.session_state.workflow_plan,
            task_df_to_text(st.session_state.task_df),
            st.session_state.rag_context,
        )

    with st.spinner("Message Agent is creating outreach drafts..."):
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
    st.session_state.goal_history = st.session_state.goal_history[:8]


init_session_state()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">AI</div>
            <div class="brand-title">AgentOS</div>
            <div class="brand-subtitle">AI Workflow Automation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    goal = st.text_area(
        "Goal",
        placeholder="Example: Apply for Google AI Internship in 7 days",
        height=126,
        key="goal_input",
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
        key="role_input",
    )

    timeline = st.selectbox(
        "Timeline",
        ["1 Day", "3 Days", "7 Days", "2 Weeks", "1 Month"],
        key="timeline_input",
    )

    company_name = st.text_input(
        "Target Company",
        placeholder="Example: Google, Microsoft, Amazon",
        key="company_input",
    )

    contact_person = st.text_input(
        "Recruiter / Contact Person",
        placeholder="Example: Hiring Manager or Recruiter Name",
        key="contact_input",
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT document",
        type=["pdf", "txt"],
        key=f"uploaded_file_{st.session_state.upload_key}",
    )

    top_k = st.slider("Evidence sections", 1, 5, key="top_k_input")

    run_button = st.button("Run Agent Workflow", type="primary")
    st.button("Clear Workspace", on_click=clear_workspace)


st.markdown(
    """
    <div class="hero-card">
        <div class="kicker">Figma-style AI Command Workspace</div>
        <div class="hero-title">Agentic AI Workflow Automation Platform</div>
        <div class="hero-subtitle">
            Plan, prioritize, automate, and generate outreach using multi-agent AI.
        </div>
        <div class="chip-row">
            <div class="chip">RAG Powered</div>
            <div class="chip">Multi-Agent</div>
            <div class="chip">Task Automation</div>
            <div class="chip">Outreach AI</div>
            <div class="chip">Report Generator</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_button:
    if not goal.strip():
        st.error("Please enter a goal first.")
    else:
        run_agent_workflow(
            goal.strip(),
            role,
            timeline,
            company_name.strip(),
            contact_person.strip(),
            uploaded_file,
            top_k,
        )
        st.success("Agent workflow completed successfully.")
        st.rerun()


total_tasks, high_priority, completed, in_progress, blocked, completion_percent = (
    get_task_metrics(st.session_state.task_df)
)
agent_status = get_agent_status()
completed_agents = sum(1 for value in agent_status.values() if value)

metric_cols = st.columns(6)

with metric_cols[0]:
    render_metric_card("📋", "Total Tasks", total_tasks, "Generated actions")
with metric_cols[1]:
    render_metric_card("📈", "Completion %", f"{completion_percent}%", "Progress tracked")
with metric_cols[2]:
    render_metric_card("🔥", "High Priority", high_priority, "Needs focus")
with metric_cols[3]:
    render_metric_card("⛔", "Blocked", blocked, "Execution risks")
with metric_cols[4]:
    render_metric_card("🧠", "RAG Score", st.session_state.rag_confidence, "Context match")
with metric_cols[5]:
    render_metric_card("⚙️", "Agents Completed", f"{completed_agents}/6", "Workflow status")

st.progress(completion_percent / 100)

st.markdown("---")
render_section_header(
    "Agent Workflow Timeline",
    "Follow the automation chain from uploaded evidence to final report.",
)
render_agent_timeline(agent_status)

st.markdown("---")

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Progress Analytics</div>', unsafe_allow_html=True)
        st.caption("Live task analytics for status, priority, and deadline load.")
        chart_a, chart_b = st.columns(2)

        with chart_a:
            st.plotly_chart(create_status_chart(st.session_state.task_df), use_container_width=True)
        with chart_b:
            st.plotly_chart(create_priority_chart(st.session_state.task_df), use_container_width=True)

        st.plotly_chart(create_deadline_chart(st.session_state.task_df), use_container_width=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Task Board Preview</div>', unsafe_allow_html=True)
        st.caption("A quick view of the generated execution board.")

        if st.session_state.task_df.empty:
            st.info("Task board preview will appear after running the agent workflow.")
        else:
            st.dataframe(st.session_state.task_df, use_container_width=True, hide_index=True)

with right:
    render_summary_card("Goal", st.session_state.current_goal or "No active goal yet")
    render_summary_card("Target Company", st.session_state.company_name or "Not selected")
    render_summary_card("Contact Person", st.session_state.contact_person or "Not selected")

    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-title">RAG Evidence Summary</div>
            <div class="evidence-box">
                <strong>RAG Confidence:</strong> {escape(str(st.session_state.rag_confidence))}<br>
                <strong>Evidence Sections:</strong> {escape(str(len(st.session_state.retrieved_chunks)))}<br>
                <strong>Document Chunks:</strong> {escape(str(len(st.session_state.document_chunks)))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-title">Next Best Action</div>
            <div class="action-box">
                <div class="eyebrow">Recommended Move</div>
                <div class="summary-value">{escape(get_next_best_action())}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        with st.container(border=True):
            st.markdown(st.session_state.workflow_plan)

        st.download_button(
            label="Download Workflow Plan",
            data=st.session_state.workflow_plan,
            file_name="workflow_plan.txt",
            mime="text/plain",
        )
    else:
        st.info("Workflow plan will appear after running the agent workflow.")

with tab2:
    st.markdown("## Editable Task Board")

    if not st.session_state.task_df.empty:
        edited_df = st.data_editor(
            st.session_state.task_df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=STATUS_OPTIONS,
                    required=True,
                ),
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=PRIORITY_OPTIONS,
                    required=True,
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
                with st.spinner("Priority Agent is re-analyzing updated task board..."):
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
        with st.container(border=True):
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
        with st.container(border=True):
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
        <div class="mini-card">
            <div class="evidence-box">
                <strong>RAG Confidence Score:</strong> {escape(str(st.session_state.rag_confidence))}<br>
                <strong>Evidence Sections:</strong> {escape(str(len(st.session_state.retrieved_chunks)))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.retrieved_chunks:
        for item in st.session_state.retrieved_chunks:
            source = escape(str(item.get("source", "Document Section")))
            score = escape(str(item.get("score", 0)))
            chunk = item.get("chunk", "")

            with st.expander(f"{source} | Similarity Score: {score}"):
                st.write(chunk)
    else:
        st.info("Relevant document evidence will appear after running with a document.")

with tab6:
    st.markdown("## Goal History")

    if st.session_state.goal_history:
        for index, item in enumerate(st.session_state.goal_history, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}.** {escape(str(item))}")
    else:
        st.info("Goal history will appear after generation.")

with tab7:
    st.markdown("## Final Report")

    if st.session_state.workflow_plan:
        final_report = build_final_report()
        st.markdown('<div class="report-shell">', unsafe_allow_html=True)
        st.markdown(final_report)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            label="Download Final Report",
            data=final_report,
            file_name="agentic_ai_workflow_automation_report.txt",
            mime="text/plain",
        )
    else:
        st.info("Final report will appear after running the workflow.")
