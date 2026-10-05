"""AgentEvals grades the path; OpenEvals grades the final answer."""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
    tool_args_match_mode="exact",
)


def reference_trajectory(question, city, expected_answer):
    # Strict trajectory matching checks message roles and tool calls/arguments,
    # not the wording of the final answer or tool response.
    return [
        HumanMessage(content=question),
        AIMessage(content="", tool_calls=[{
            "name": "get_weather", "args": {"city": city}, "id": "reference_call",
        }]),
        ToolMessage(content="fixture result", tool_call_id="reference_call"),
        AIMessage(content=expected_answer),
    ]


def make_answer_evaluator(model=None, judge=None):
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="answer_correctness",
        model=model,
        judge=judge,
    )


def final_answer(result):
    # Handles string output and providers returning text content blocks.
    return result["messages"][-1].text
