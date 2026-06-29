"""Streamlit entry point for the Agentic AI Workflow Automation Platform."""

import importlib
import sys


if "utils.app" in sys.modules:
    importlib.reload(sys.modules["utils.app"])
else:
    import utils.app  # noqa: F401
