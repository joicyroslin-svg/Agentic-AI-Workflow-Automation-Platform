import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "20000"))


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

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        return f"Gemini library failed to load: {exc}"

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    errors = []

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception as exc:
            errors.append(f"{model}: {exc}")
            continue

    return (
        "AI response failed. Please check your API key or internet connection.\n\n"
        + "\n".join(errors[-3:])
    )


def planner_agent(goal, role, timeline):
    prompt = f"""
You are a Planner Agent in an Agentic AI Workflow Automation Platform.

User Goal:
{goal}

User Role:
{role}

Timeline:
{timeline}

Create a clear workflow plan.

Format:

## Goal Understanding
Explain the goal in simple words.

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

Keep it simple, practical, and useful for a student or fresher.
"""

    return ask_gemini(prompt)


def task_agent(goal, role, timeline, workflow_plan):
    prompt = f"""
You are a Task Agent in an Agentic AI Workflow Automation Platform.

User Goal:
{goal}

User Role:
{role}

Timeline:
{timeline}

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
- Do not add extra explanation before or after the table

Example:
Update resume for AI role | High | Not Started | Day 1
Find 10 internship openings | High | Not Started | Day 1
Write recruiter message | Medium | Not Started | Day 2
"""

    return ask_gemini(prompt)
