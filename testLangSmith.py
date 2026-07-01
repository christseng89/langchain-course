import os
import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = anthropic.Anthropic()

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
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    result = run_weather_agent("What's the weather like in San Francisco and Tokyo?")
    print(result)
