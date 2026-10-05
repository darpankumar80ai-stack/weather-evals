"""One parametrized test demonstrates all four tools together."""
import os
import pytest
from langsmith import testing as t
from agent import build_agent, ask
from evaluators import (
    trajectory_evaluator, reference_trajectory, make_answer_evaluator, final_answer,
)

CASES = [
    ("What is the weather in Tokyo?", "Tokyo", "Tokyo is sunny and 22 degrees Celsius."),
    ("What is the weather in London? Should I bring an umbrella?", "London",
     "London is rainy and 12 degrees Celsius. Bring an umbrella."),
    ("What is the weather in Atlantis?", "Atlantis", "Weather data for Atlantis is unavailable."),
]


# 1. PYTEST runs this test once per example.
# 2. LANGSMITH records examples, runs, traces, feedback, and pass/fail.
@pytest.mark.live
@pytest.mark.langsmith(test_suite_name="Weather agent demo")
@pytest.mark.parametrize("question,city,expected_answer", CASES)
def test_weather_agent(question, city, expected_answer):
    t.log_inputs({"question": question, "city": city})
    t.log_reference_outputs({"answer": expected_answer})

    result = ask(build_agent(), question)
    answer = final_answer(result)
    t.log_outputs({"answer": answer})

    with t.trace_feedback():
        # 3. AGENTEVALS checks one weather call with the correct city.
        path_score = trajectory_evaluator(
            outputs=result["messages"],
            reference_outputs=reference_trajectory(question, city, expected_answer),
        )
        t.log_feedback(key=path_score["key"], score=path_score["score"])

        # 4. OPENEVALS asks a judge model to grade the final answer.
        grader = make_answer_evaluator(model=os.environ["JUDGE_MODEL"])
        answer_score = grader(
            inputs=question, outputs=answer, reference_outputs=expected_answer,
        )
        t.log_feedback(
            key=answer_score["key"], score=answer_score["score"],
            comment=answer_score.get("comment"),
        )

    # Pytest enforces both scores; LangSmith records the pass/fail result.
    assert path_score["score"] is True, path_score
    assert answer_score["score"] is True, answer_score
