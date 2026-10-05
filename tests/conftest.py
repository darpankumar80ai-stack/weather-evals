import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Existing environment values take precedence. Secrets are never printed.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        help="Run paid model calls and LangSmith tracking",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        required = ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "AGENT_MODEL", "JUDGE_MODEL"]
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise pytest.UsageError("Live mode requires: " + ", ".join(missing))
        if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
            raise pytest.UsageError(
                "Set LANGSMITH_TRACING=true for the tracked live example"
            )
    else:
        for item in items:
            if item.get_closest_marker("live"):
                item.add_marker(
                    pytest.mark.skip(reason="Use --live with configured credentials")
                )


def pytest_configure(config):
    # Loading .env must not enable cloud tracing for offline checks.
    if not config.getoption("--live"):
        os.environ["LANGSMITH_TRACING"] = "false"
