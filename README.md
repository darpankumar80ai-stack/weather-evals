# Weather agent: four evaluation tools together

This small Python project answers weather questions using a tool. Weather is a
fixed fixture so evaluations have known answers. It is not a live weather service.

The cases are Tokyo (sunny, 22°C), London (rainy, 12°C, bring an umbrella), and
Atlantis (data unavailable; do not invent a forecast).

## Which tool does what?

| Tool | Where to look | Job |
|---|---|---|
| pytest | `tests/test_live_evals.py` | Runs each example and asserts both grading scores pass. |
| LangSmith | `@pytest.mark.langsmith`, `t.log_*`, `t.trace_feedback()` | Tracks test examples, agent traces, grading feedback and experiment results. |
| AgentEvals | `evaluators.py`: `trajectory_evaluator` | Checks that the agent calls `get_weather` once with the correct city. |
| OpenEvals | `evaluators.py`: `make_answer_evaluator` | Uses a judge model to compare the final answer with a reference answer. |

The flow is:

```text
pytest selects an example
  → LangChain agent calls the fixture weather tool
  → AgentEvals grades the tool trajectory
  → OpenEvals grades the final answer
  → LangSmith records traces and scores
  → pytest assertions determine pass/fail
```

LangChain creates the agent; it is the application being tested. The other four
tools support testing and evaluation. AgentEvals and OpenEvals can be used without
LangSmith. pytest can also run entirely offline tests.

## Set up

Use Python 3.12 (the version used for local validation). In this project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the direct packages used for validation. It is not a full
transitive dependency lock.

## First run: offline checks

```sh
pytest -q
```

No keys or accounts are needed. Six checks use scripted models to verify real
agent/tool wiring, rejection of wrong-city/missing/duplicate calls, and propagation
of both passing and failing OpenEvals judge results. The three live cases are skipped.

These checks do **not** establish that a real model follows the instructions or
that a real judge correctly assesses quality. They do not test uploading to LangSmith.

## Run all four tools with real models and LangSmith

```sh
cp .env.example .env
```

Edit `.env` locally and fill in:

- `OPENAI_API_KEY`: your OpenAI model credentials.
- `LANGSMITH_API_KEY`: your LangSmith credentials.
- `AGENT_MODEL`: an OpenAI model that supports tool calling, written as `openai:MODEL_NAME`.
- `JUDGE_MODEL`: an OpenAI model that supports structured output, written as `openai:MODEL_NAME`.
- Keep `LANGSMITH_TRACING=true` to capture the agent's internal calls.
- Set `LANGSMITH_WORKSPACE_ID` if your key needs a specific workspace.

Both model fields may use the same model for this learning example. In a serious
evaluation, validate your judge against human ratings. Keys stay in your local
environment or `.env`; `.env` is ignored by Git.

```sh
pytest --live -q tests/test_live_evals.py
```

This makes paid model calls and sends the example inputs, outputs, traces and
scores to your LangSmith workspace. The fixture contains no personal information.
The default flow uses only one tool and needs no network weather API.

In LangSmith, look for the **Weather agent demo** dataset/test suite and its latest
experiment. Each row should contain the final answer, trajectory score,
`answer_correctness` score and pytest pass/fail status. The judge's explanation is
logged as feedback; grading calls are separated from application calls.

## Read the central test

1. `@pytest.mark.parametrize` supplies the three input/reference pairs.
2. `@pytest.mark.langsmith` enables tracking of the test.
3. `ask(build_agent(), question)` runs the real model and weather tool.
4. `trajectory_evaluator(...)` compares the messages/tool calls with the reference path.
5. `grader(...)` uses OpenEvals and a judge model to evaluate the final answer.
6. `t.log_feedback(...)` records both scores in LangSmith.
7. The final assertions fail the test if either evaluator fails.

Strict trajectory matching here checks message structure, tool names, ordering and
arguments. It does not enforce exact final-answer wording. This project deliberately
requires exactly one weather call; a larger workflow may need looser trajectory rules.

A correct tool call can still produce a wrong answer, so both evaluations matter.
Conversely, an agent may guess a correct answer without using the required tool.

## Iterate

Change the agent instructions in `agent.py` and rerun the same live cases. Compare
experiments in LangSmith. Repeat trials to see variability; keep adding realistic
and edge cases. Three examples are a teaching dataset, not a production benchmark.
An evaluator score does not prove correctness, and a model judge can be wrong.

## Validation performed here

Local run: **6 passed, 3 skipped**. Live model calls and LangSmith uploads were not
run because this example was prepared without configured project credentials.

## Primary documentation

- [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html)
- [LangSmith pytest integration](https://docs.langchain.com/langsmith/pytest)
- [AgentEvals trajectory evaluators](https://docs.langchain.com/langsmith/trajectory-evals)
- [OpenEvals Python documentation](https://github.com/langchain-ai/openevals/tree/main/python)

OpenEvals in this project is LangChain's library, distinct from OpenAI's Evals product.
