"""Tiny weather agent. Weather is a fixed test fixture, not a live forecast."""
import os
from langchain.agents import create_agent
from langchain.tools import tool

WEATHER = {
    "Tokyo": {"temperature_c": 22, "condition": "sunny"},
    "London": {"temperature_c": 12, "condition": "rainy"},
}


@tool
def get_weather(city: str) -> dict:
    """Get fixture weather for a city, using its canonical English name."""
    if city not in WEATHER:
        return {"city": city, "available": False}
    return {"city": city, "available": True, **WEATHER[city]}


def build_agent(model=None):
    return create_agent(
        model=model if model is not None else os.environ["AGENT_MODEL"],
        tools=[get_weather],
        system_prompt=(
            "Answer weather questions using get_weather exactly once. "
            "Use canonical English city names. Report the city, temperature "
            "in Celsius, and condition returned by the tool. Recommend an "
            "umbrella when it is rainy. If data is unavailable, say so; "
            "do not invent a forecast. These are demo weather fixtures."
        ),
    )


def ask(agent, question):
    return agent.invoke({"messages": [{"role": "user", "content": question}]})
