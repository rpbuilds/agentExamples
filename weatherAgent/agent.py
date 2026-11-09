"""
Weather agent: asks Claude what to wear, with live weather from Open-Meteo.
See README.md for setup. Quick start:

    pip install -r requirements.txt
    python agent.py --help

Open-Meteo needs no API key. Docs: https://open-meteo.com/en/docs
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import anthropic
import requests

# swap model as needed, haiku is fine and cheap
MODEL: str = "claude-haiku-5-5"
MAX_STEPS: int = 8          # hard cap. never run an agent loop unbounded.
AUSTIN: tuple[float, float] = (30.27, -97.74)   # (latitude, longitude)


client: anthropic.Anthropic = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# TOOL SCHEMAS and DEFINITIONS
#
# The description is a prompt. It is the model's only information about when to
# reach for this tool. Say when to use it, when NOT to, and what comes back.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_current_weather",
        "description": (
            "Get weather conditions RIGHT NOW at a location. "
            "Use when the question is about going outside immediately. "
            "Do not use for anything more than ~30 minutes out — use get_forecast instead. "
            "Returns: temperature in Fahrenheit, wind speed in miles per hour, "
            "and a short text condition such as 'light rain'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Decimal degrees, e.g. 30.27"},
                "longitude": {"type": "number", "description": "Decimal degrees, e.g. -97.74"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "get_forecast",
        "description": (
            "Get hour-by-hour forecast for the next N hours at a location. "
            "Use when the question concerns a time later today, or when "
            "conditions may change between now and when the person is outside. "
            "Returns: one row per hour with temperature in Fahrenheit, "
            "precipitation probability as a percentage, and wind speed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "hours": {
                    "type": "integer",
                    "description": "How many hours ahead, 1 to 24.",
                },
            },
            "required": ["latitude", "longitude", "hours"],
        },
    },
    {
        "name": "get_todays_events",
        "description": (
            "Get today's calendar events, to find out WHEN the person will be "
            "outside and for how long. Use when the timing of their exposure "
            "matters and they haven't stated it. Takes no arguments. "
            "Returns: a list of events with start time, end time, title, and "
            "whether the event is outdoors."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# TOOL BODIES
#
# Each returns a STRING. The model reads whatever is returned here
# Editing the returned text has a real impact on the final answer the model returns
# ---------------------------------------------------------------------------

WEATHER_CODES: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}

def get_current_weather(latitude: float, longitude: float) -> str:
    resp: requests.Response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        },
        timeout=10,
    )
    resp.raise_for_status()   # turns an HTTP error into an exception
    data: dict[str, Any] = resp.json()
    curr_cond: dict[str, Any] = data["current"]

    return (
        f"{curr_cond['temperature_2m']}F, {curr_cond['precipitation']} inches of "
        f"precipitation, weather code {curr_cond['weather_code']}, wind "
        f"{curr_cond['wind_speed_10m']} mph"
    )



def get_forecast(latitude: float, longitude: float, hours: int) -> str:
    hours = max(1, min(int(hours), 24))   # the model might ask for 0 or 100

    resp: requests.Response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
            "forecast_days": 2,              # so "next 24 hours" never runs off the end
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",              # times come back in the location's local time
        },
        timeout=10,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    # Each key ("time", "temperature_2m", ...) maps to a list, one entry per hour.
    hourly: dict[str, list[Any]] = data["hourly"]

    # hourly["time"] starts at midnight today, not now. Find the current hour.
    # Times look like "2026-11-08T14:00", so plain string comparison works.
    local_now: datetime = datetime.now(timezone.utc) + timedelta(seconds=data["utc_offset_seconds"])
    now_str: str = local_now.strftime("%Y-%m-%dT%H:00")
    start: int = next((i for i, t in enumerate(hourly["time"]) if t >= now_str), 0)

    lines: list[str] = []
    for i in range(start, min(start + hours, len(hourly["time"]))):
        clock: str = hourly["time"][i][11:16]     # "2026-11-08T14:00" -> "14:00"
        lines.append(
            f"{clock}: {hourly['temperature_2m'][i]:.0f}F, "
            f"{hourly['precipitation_probability'][i]}% chance of rain, "
            f"wind {hourly['wind_speed_10m'][i]:.0f} mph"
        )
    return "\n".join(lines)


def get_todays_events() -> str:
    # Fake calendar. Not from Open-Meteo: a weather API knows nothing about your schedule.
    # TODO: Swap in Google Calendar later.
    # Each event: (start time, end time, title, is_outdoors)
    events: list[tuple[str, str, str, bool]] = [
        ("08:30", "09:00", "Bike commute to office", True),
        ("12:00", "13:00", "Lunch with Sam (restaurant patio)", True),
        ("14:00", "15:00", "Team meeting", False),
        ("17:30", "18:00", "Bike commute home", True),
    ]
    return "\n".join(
        f"{start}-{end}: {title} ({'outdoors' if outside else 'indoors'})"
        for start, end, title, outside in events
    )


# Maps a tool name to the Python function that runs it. Callable[..., str]
#   means "a function taking any arguments that returns a str".
RUN: dict[str, Callable[..., str]] = {
    "get_current_weather": get_current_weather,
    "get_forecast": get_forecast,
    "get_todays_events": get_todays_events,
}

# System prompt: standing instructions (role, rules, answer style) sent to Claude
# on every API call. It's separate from `messages`, which holds the conversation.
SYSTEM_PROMPT: str = (
    f"You help someone in Austin, Texas (latitude {AUSTIN[0]}, longitude "
    f"{AUSTIN[1]}) decide what to wear and when to go outside. Use the tools "
    "to find out what you need. Do not guess at weather or schedule — look it "
    "up. When you have enough, answer in two or three sentences and say what "
    "you based it on."
)


# ---------------------------------------------------------------------------
# THE AGENT LOOP
#
# This is the entire concept. Everything else in the agent world is a wrapper
# around these twenty lines.
# ---------------------------------------------------------------------------

def to_pretty_json(obj: Any) -> str:
    """Pretty-print anything in the conversation as indented JSON.

    default= handles the SDK objects in Claude's replies (TextBlock,
    ToolUseBlock, ...), which json can't serialize on its own: they're turned
    into plain dicts first.
    """
    return json.dumps(
        obj,
        indent=2,
        default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o),
    )


def run_agent(
    question: str,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Answer one question, calling tools as Claude asks for them.

    verbose: print a short line per step and per tool call.
    debug:   also print the full JSON sent to and received from Claude on every
             call, plus each tool's full output. Long. Implies verbose.
    """
    verbose = verbose or debug
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for step in range(MAX_STEPS):
        if debug:
            print(f"\n[step {step + 1}] calling {MODEL} with messages:\n{to_pretty_json(messages)}")

        response: anthropic.types.Message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if debug:
            # The whole reply: content blocks, stop_reason, and token usage.
            # Parallel tool calls show up as several tool_use blocks in
            # "content", each with its own id.
            print(f"\n[step {step + 1}] {MODEL} responded with:\n{to_pretty_json(response)}")
        elif verbose:
            print(
                f"[step {step + 1}] stop_reason={response.stop_reason}, "
                f"tokens in={response.usage.input_tokens} out={response.usage.output_tokens}"
            )

        # Append the model's FULL response, tool_use blocks included.
        # Dropping them breaks the exchange — common first bug.
        messages.append({"role": "assistant", "content": response.content})

        # return the final results if the model doesn't need to call anymore tools
        if response.stop_reason != "tool_use":
            final: str = "".join(b.text for b in response.content if b.type == "text")
            if verbose:
                print(f"[done in {step + 1} model calls]")
            return final

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"[step {step + 1} toolCall] -> {block.name}({json.dumps(block.input)})")

            output: str = ""
            is_error: bool = False
            try:
                output = RUN[block.name](**block.input)
                is_error = False
            except Exception as exc:
                # Hand errors back to the model rather than crashing. It can
                # often recover — retry, or try a different tool.
                output = f"{type(exc).__name__}: {exc}"
                is_error = True
            if debug:
                label: str = "ERROR" if is_error else "result"
                print(f"[step {step + 1} tool {label}] {block.name}:\n{output}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,     # must match the request's id
                "content": output,
                "is_error": is_error,
            })

        # Tool results go back as a user message, nothing else in it.
        messages.append({"role": "user", "content": tool_results})

    return f"Hit the step limit of {MAX_STEPS} without finishing."


# ---------------------------------------------------------------------------
# COMMAND LINE
#
#   python agent.py                         run the sample QUESTIONS below
#   python agent.py "Do I need a jacket?"   ask your own question
#   python agent.py -v ...                  show each step and tool call
#   python agent.py -d ...                  show full JSON for every call (long)
#   python agent.py -h                      list all options
# ---------------------------------------------------------------------------

QUESTIONS: list[str] = [
    "Do I need a jacket if I walk out right now?",
    #"Should I move my run to this morning or wait until evening?",
    #"I'm outside at some point today — what should I wear?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Claude what to wear, using live weather from Open-Meteo.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask. Leave it out to run the sample questions.",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show a short line for each step and each tool call.",
    )
    output_mode.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Show the full JSON sent to and received from Claude on every "
             "call, plus each tool's full output. Very long.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")

    # Words are joined, so quotes around the question are optional.
    questions: list[str] = [" ".join(args.question)] if args.question else QUESTIONS
    for q in questions:
        print(f"\n=== {q}")
        print(run_agent(q, verbose=args.verbose, debug=args.debug))