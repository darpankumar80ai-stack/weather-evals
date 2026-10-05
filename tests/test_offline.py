"""No credentials, network requests, or model charges are needed here.

Scripted models test the wiring, not the intelligence of a real model.
"""
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from agent import build_agent, ask
from evaluators import (
    trajectory_evaluator, reference_trajectory, make_answer_evaluator, final_answer,
)


class ScriptedToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_executes_weather_tool():
    model = ScriptedToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "get_weather", "args": {"city": "Tokyo"}, "id": "actual_call",
        }]),
        AIMessage(content="Tokyo is sunny and 22 degrees Celsius."),
    ])
    result = ask(build_agent(model), "Weather in Tokyo?")
    assert '"temperature_c": 22' in result["messages"][2].content
    score = trajectory_evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory("Weather in Tokyo?", "Tokyo", "reference"),
    )
    assert score["score"] is True
    assert final_answer(result) == "Tokyo is sunny and 22 degrees Celsius."


@pytest.mark.parametrize("defect", ["wrong_city", "missing_tool", "duplicate_call"])
def test_trajectory_rejects_bad_paths(defect):
    reference = reference_trajectory("Weather in Tokyo?", "Tokyo", "reference")
    if defect == "wrong_city":
        actual = reference_trajectory("Weather in Tokyo?", "London", "reference")
    elif defect == "missing_tool":
        actual = [reference[0], reference[-1]]
    else:
        actual = reference[:-1] + reference[1:3] + reference[-1:]
    score = trajectory_evaluator(outputs=actual, reference_outputs=reference)
    assert score["score"] is False


class ScriptedJudge(FakeMessagesListChatModel):
    verdict: bool

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: {
            "score": self.verdict,
            "reasoning": "Scripted verdict for an offline integration check.",
        })


@pytest.mark.parametrize("verdict", [True, False])
def test_openevals_propagates_judge_verdict(verdict):
    grader = make_answer_evaluator(judge=ScriptedJudge(responses=[], verdict=verdict))
    result = grader(inputs="Weather?", outputs="Example answer", reference_outputs="Reference")
    assert result["key"] == "answer_correctness"
    assert result["score"] is verdict
