# Weather Agent

A small AI agent written to answers questions like
*"Do I need a jacket if I walk out right now?"* using live weather data and a
(fake currently) daily calendar.

## Goal

The point is to show what makes an agent different from a script. A question
like "what should I wear today?" has no fixed path to an answer. Sometimes the
current weather is enough. Sometimes the forecast matters. Sometimes it depends
on *when* you'll be outside. The code never says which of these to check or in
what order. Claude decides, based on the question and the tool descriptions.

A second goal is to build, from scratch, the simple harness that sits around a
base model, in order to understand at a basic level what chat products like
ChatGPT and the Claude app do behind the scenes. Those products wrap the model
in the same pieces this project has (a system prompt, tools, and a loop that
runs tool calls and feeds the results back), just with many more tools and a
polished interface.

## How it works

`agent.py` has four parts:

1. **Tool schemas (`TOOLS`).** A name, description, and input format for each
   tool. Claude only ever sees these descriptions, never the Python code, so
   the descriptions decide when Claude reaches for each tool.
2. **Tool functions.** The Python code that actually runs when Claude asks for
   a tool:
   - `get_current_weather`: conditions right now, from Open-Meteo.
   - `get_forecast`: hour-by-hour forecast for the next N hours, from Open-Meteo.
   - `get_todays_events`: a hardcoded list of today's events, some outdoors.
3. **System prompt (`SYSTEM_PROMPT`).** Standing instructions sent on every
   call: who Claude is helping, where they are, and "don't guess, look it up."
4. **The agent loop (`run_agent`).** Each pass through the loop:
   1. Send the conversation so far to Claude.
   2. If Claude replies with tool requests (`stop_reason == "tool_use"`), run
      those tools, add the results to the conversation, and loop again.
   3. Otherwise, Claude has its answer, and the loop returns it.

   The loop stops after `MAX_STEPS` calls no matter what, so it can't run
   forever.

The Claude API keeps no memory between calls. The `messages` list *is* the
conversation, and the whole list is resent on every call.

## Open-Meteo

[Open-Meteo](https://open-meteo.com/) is a free, open-source weather API. It
needs no API key or account for non-commercial use. The agent calls its
forecast endpoint (`https://api.open-meteo.com/v1/forecast`) with a latitude
and longitude, and asks for either `current` conditions or `hourly` data.

Docs, with a tool for building request URLs: <https://open-meteo.com/en/docs>

## Getting started

### 1. Requirements

- Python 3.10 or newer
- An Anthropic API key from <https://platform.claude.com> (Settings → API
  Keys). The API is billed separately from a Claude.ai subscription and needs
  prepaid credits. With the default Haiku model, a run costs well under a cent.

### 2. Install the dependencies

From this folder:

```bash
pip install -r requirements.txt
```

Optionally, use a virtual environment first so the packages stay local to this
project:

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows PowerShell
source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

### 3. Set your API key

Windows PowerShell, current terminal only:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Windows, permanently (then fully restart VS Code or your terminal):

```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

macOS / Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Never commit your key to git.

### 4. Run it

On Windows, use `python`, not `python3`.

```bash
python agent.py                                   # run the sample questions
python agent.py "Do I need a jacket right now?"   # ask your own question
python agent.py -h                                # show all options
```

## Output modes

| Command | What you see |
|---|---|
| `python agent.py "..."` | Just the final answer. |
| `python agent.py -v "..."` | **Verbose:** one line per model call (stop reason, token counts) and one line per tool call. |
| `python agent.py -d "..."` | **Debug:** everything in verbose mode, plus the full JSON conversation sent to Claude on every call, Claude's full JSON response, and each tool's full output. Very long, but it shows exactly what goes back and forth. |

`-v` and `-d` can't be combined. Debug already includes everything verbose
shows.

## Limitations and next steps

This is a deliberately minimal agen. It works, but a real agent
would need more. Things to improve:

- **Evals.** There's no way to measure whether a change (a new prompt, tool
  description, or model) made answers better or worse. An eval is a fixed set of
  test questions with expected outcomes (for example, "a question about 'right
  now' should call `get_current_weather` and nothing else") that you rerun after
  every change.
- **Real data.** `get_todays_events` is hardcoded. Connecting it to a real
  calendar, such as the Google Calendar API, would make the agent useful.
- **More tools.** For example, geocoding (turning a city name into
  coordinates, so the agent isn't tied to Austin), or actions such as sending a
  reminder.
- **Conversation memory.** Each question starts from scratch. A chat loop that
  keeps `messages` between questions would allow follow-ups like "what about
  tomorrow?"
- **Error handling.** API failures, rate limits, and a `max_tokens` cutoff
  (currently treated as a finished answer) should be handled explicitly.
- **Cost and speed.** Prompt caching would avoid reprocessing the same system
  prompt and tool descriptions on every call, and streaming would show the
  answer as it's written.
- **Safety.** Tools that take actions (send, delete, buy) would need limits and
  a confirmation step before they run.

## Things to try

- Run the three sample questions (uncomment them in `QUESTIONS`) with `-v` and
  compare which tools Claude calls for each. They should differ.
- Reword a tool description and see whether Claude's choices change.
- Change what a tool returns (for example, raw JSON instead of a sentence) and
  see how the answers change.
- Use `-d` to watch the `messages` list grow with each call.
