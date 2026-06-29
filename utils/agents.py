import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None


def ask_gemini(prompt):
    api_key = get_api_key()

    if not api_key:
        return "Gemini API key is missing. Please add GEMINI_API_KEY in your .env file."

    client = genai.Client(api_key=api_key)

    models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash"
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception:
            continue

    return "AI response failed. Please check your API key or internet connection."


def planner_agent(goal, role, timeline, rag_context=""):
    prompt = f"""
You are a Planner Agent in an Agentic AI Workflow Automation Platform.

User Goal:
{goal}

User Role:
{role}

Timeline:
{timeline}

Retrieved Document Context:
{rag_context}

Create a clear workflow plan.

Rules:
- If document context is available, use it while planning.
- If document context is missing, create a general workflow plan.
- Keep the output practical for a student or fresher.
- Do not create fake information.

Format:

## Goal Understanding
Explain the goal in simple words.

## Context Used
Mention whether uploaded document context was used or not.

## Main Objective
What should the user achieve?

## Task Breakdown
Create 8 to 10 clear tasks.

## Priority Plan
Divide tasks into:
- High Priority
- Medium Priority
- Low Priority

## Suggested Timeline
Give a simple day-wise or step-wise timeline.

## Tools Needed
Suggest useful tools.

## Possible Risks
Mention what can go wrong.

## Next Best Action
Tell the user what to do first.
"""

    return ask_gemini(prompt)


def task_agent(goal, role, timeline, workflow_plan, rag_context=""):
    prompt = f"""
You are a Task Agent in an Agentic AI Workflow Automation Platform.

User Goal:
{goal}

User Role:
{role}

Timeline:
{timeline}

Retrieved Document Context:
{rag_context}

Planner Agent Output:
{workflow_plan}

Convert the workflow plan into a clean task checklist.

Very important:
Return tasks only in this exact format:

Task | Priority | Status | Deadline

Rules:
- Create 8 to 10 tasks
- Priority must be High, Medium, or Low
- Status must be Not Started
- Deadline should be simple, like Day 1, Day 2, Day 3, Week 1, etc.
- Use document context if available
- Do not add extra explanation before or after the table

Example:
Update resume for AI role | High | Not Started | Day 1
Find 10 internship openings | High | Not Started | Day 1
Write recruiter message | Medium | Not Started | Day 2
"""

    return ask_gemini(prompt)


def priority_agent(goal, role, timeline, workflow_plan, task_table_text, rag_context=""):
    prompt = f"""
You are a Priority Agent in an Agentic AI Workflow Automation Platform.

User Goal:
{goal}

User Role:
{role}

Timeline:
{timeline}

Retrieved Document Context:
{rag_context}

Workflow Plan:
{workflow_plan}

Current Task Checklist:
{task_table_text}

Analyze the current tasks and give priority guidance.

Format:

## Priority Summary
Explain which tasks matter most.

## Critical Tasks
List the top 3 tasks the user should do first.

## Blocker Risk
Mention tasks that can block progress if delayed.

## Time Management Plan
Give a simple plan based on the selected timeline.

## Completion Strategy
Explain how to finish the workflow effectively.

## Next Best Action
Tell the user exactly what to do next.

Keep it simple, practical, and useful for a student or fresher.
"""

    return ask_gemini(prompt)