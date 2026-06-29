import pandas as pd
import streamlit as st

from utils.agents import planner_agent, task_agent


TASK_COLUMNS = ["Task", "Priority", "Status", "Deadline"]


st.set_page_config(
    page_title="Agentic AI Workflow Automation Platform",
    layout="wide",
)


def init_session_state():
    defaults = {
        "workflow_plan": "",
        "task_text": "",
        "task_df": pd.DataFrame(columns=TASK_COLUMNS),
        "goal_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_task_text(task_text):
    tasks = []

    for line in task_text.splitlines():
        clean_line = line.strip()

        if not clean_line or "|" not in clean_line:
            continue

        if "Task | Priority | Status | Deadline" in clean_line:
            continue

        parts = [part.strip() for part in clean_line.split("|")]

        if len(parts) >= 4:
            tasks.append(
                {
                    "Task": parts[0],
                    "Priority": parts[1],
                    "Status": parts[2],
                    "Deadline": parts[3],
                }
            )

    return pd.DataFrame(tasks, columns=TASK_COLUMNS)


def get_task_metrics(task_df):
    total_tasks = len(task_df)

    if total_tasks == 0:
        return 0, 0, 0, 0

    high_priority = len(task_df[task_df["Priority"] == "High"])
    completed = len(task_df[task_df["Status"] == "Completed"])
    in_progress = len(task_df[task_df["Status"] == "In Progress"])

    return total_tasks, high_priority, completed, in_progress


def clear_workspace():
    st.session_state.workflow_plan = ""
    st.session_state.task_text = ""
    st.session_state.task_df = pd.DataFrame(columns=TASK_COLUMNS)


def run_generation(goal, role, timeline):
    with st.spinner("Planner Agent is creating workflow plan..."):
        st.session_state.workflow_plan = planner_agent(goal, role, timeline)

    with st.spinner("Task Agent is creating task checklist..."):
        st.session_state.task_text = task_agent(
            goal,
            role,
            timeline,
            st.session_state.workflow_plan,
        )

    st.session_state.task_df = parse_task_text(st.session_state.task_text)
    st.session_state.goal_history.insert(0, goal)
    st.session_state.goal_history = st.session_state.goal_history[:5]


def render_dashboard():
    total_tasks, high_priority, completed, in_progress = get_task_metrics(
        st.session_state.task_df
    )

    st.title("Agentic AI Workflow Automation Platform")
    st.caption("Convert goals into workflow plans and AI-generated task checklists.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Tasks", total_tasks)
    metric_cols[1].metric("High Priority", high_priority)
    metric_cols[2].metric("In Progress", in_progress)
    metric_cols[3].metric("Completed", completed)


def render_workspace():
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.subheader("Goal Workspace")

        with st.form("goal_form", clear_on_submit=False):
            goal = st.text_area(
                "Enter your goal",
                placeholder="Example: Prepare and apply for Google AI Internship in 7 days",
                height=140,
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
                    "Startup Founder",
                ],
            )

            timeline = st.selectbox(
                "Timeline",
                [
                    "1 Day",
                    "3 Days",
                    "7 Days",
                    "2 Weeks",
                    "1 Month",
                ],
            )

            submitted = st.form_submit_button("Generate Workflow + Tasks")

        if submitted:
            if not goal.strip():
                st.error("Please enter a goal first.")
            else:
                run_generation(goal, role, timeline)
                st.success("Workflow plan and task checklist generated successfully.")

        if st.button("Clear Workspace"):
            clear_workspace()
            st.success("Workspace cleared.")
            st.rerun()

    with right:
        st.subheader("Task Analytics")

        if st.session_state.task_df.empty:
            st.info("Analytics will appear after tasks are generated.")
            return

        status_counts = st.session_state.task_df["Status"].value_counts()
        priority_counts = st.session_state.task_df["Priority"].value_counts()

        st.write("Task status")
        st.bar_chart(status_counts)

        st.write("Task priority")
        st.bar_chart(priority_counts)


def render_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Workflow Plan", "Task Checklist", "Goal History", "Project Notes"]
    )

    with tab1:
        st.subheader("AI Workflow Plan")

        if st.session_state.workflow_plan:
            st.markdown(st.session_state.workflow_plan)
            st.download_button(
                label="Download Workflow Plan",
                data=st.session_state.workflow_plan,
                file_name="workflow_plan.txt",
                mime="text/plain",
            )
        else:
            st.info("Workflow plan will appear here.")

    with tab2:
        st.subheader("Editable Task Checklist")

        if st.session_state.task_df.empty:
            st.info("Task checklist will appear here after generation.")
            return

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

        st.download_button(
            label="Download Task Checklist CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="workflow_tasks.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Goal History")

        if st.session_state.goal_history:
            for index, item in enumerate(st.session_state.goal_history, start=1):
                st.write(f"{index}. {item}")
        else:
            st.info("Goal history will appear here.")

    with tab4:
        st.subheader("Project Notes")
        st.write(
            """
            This platform uses a Planner Agent to create a workflow plan and a
            Task Agent to convert that plan into a structured checklist.
            """
        )


def main():
    init_session_state()
    render_dashboard()
    st.divider()
    render_workspace()
    st.divider()
    render_tabs()


def run_app():
    try:
        main()
    except Exception as exc:
        st.error("The app crashed while rendering. Details are shown below.")
        st.exception(exc)


if __name__ == "__main__":
    run_app()
