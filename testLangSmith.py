import os
import time
import uuid

import anthropic
from dotenv import load_dotenv
from langsmith import Client, traceable

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = anthropic.Anthropic()
langsmith_client = Client()
PROJECT_NAME = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT", "default")

tools = [
  {
    "name": "get_weather",
    "description": "Gets the current weather for a given city",
    "input_schema": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "The city to get weather for",
        }
      },
      "required": ["city"],
    },
  }
]

WEATHER_DATA = {
  "San Francisco": "Foggy, 62°F",
  "New York": "Sunny, 75°F",
  "London": "Rainy, 55°F",
  "Tokyo": "Clear, 68°F",
  "Taipei": "Cloudy, 80°F",
  "Nanjing": "Sunny, 85°F",
}


def get_weather(city: str) -> str:
  return WEATHER_DATA.get(city, "Weather data not available")


@traceable(name="weather-agent")
def run_weather_agent(query: str) -> str:
  messages = [{"role": "user", "content": query}]

  while True:
    response = client.messages.create(
      model="claude-opus-4-8",
      max_tokens=1024,
      system="You are a friendly travel assistant who helps with weather information.",
      tools=tools,
      messages=messages,
    )

    if response.stop_reason == "end_turn":
      return next(b.text for b in response.content if b.type == "text")

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
      if block.type == "tool_use":
        result = get_weather(**block.input) if block.name == "get_weather" else "Unknown tool"
        tool_results.append(
          {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
          }
        )

    messages.append({"role": "user", "content": tool_results})


def print_trace_runs(trace_id: uuid.UUID, timeout: float = 15.0) -> None:
  """Fetch and print every run LangSmith captured for a trace.

  Ingestion is asynchronous, so a just-finished trace may not be queryable
  yet -- flush the pending run and poll briefly until it shows up.
  """
  langsmith_client.flush()

  deadline = time.monotonic() + timeout
  runs = []
  while time.monotonic() < deadline:
    runs = list(langsmith_client.list_runs(project_name=PROJECT_NAME, trace_id=trace_id))
    if runs and all(run.end_time is not None for run in runs):
      break
    time.sleep(1)

  if not runs:
    print(f"No runs found for trace {trace_id} (project={PROJECT_NAME}).")
    return

  runs.sort(key=lambda r: r.start_time)
  print(f"\nLangSmith trace {trace_id} — {len(runs)} run(s):")
  for run in runs:
    latency = f"{(run.end_time - run.start_time).total_seconds():.2f}s" if run.end_time else "n/a"
    print(
      f"  - {run.name:<20} type={run.run_type:<10} status={run.status} "
      f"latency={latency} tokens={run.total_tokens} error={run.error}"
    )


if __name__ == "__main__":
  trace_id = uuid.uuid4()
  result = run_weather_agent(
    "What's the weather like in Taipei, Nanjing, San Francisco and Tokyo?",
    langsmith_extra={"run_id": trace_id, "client": langsmith_client},
  )
  print(result)

  print("\n⏳ Waiting for LangSmith to ingest the trace...")
  print_trace_runs(trace_id)
