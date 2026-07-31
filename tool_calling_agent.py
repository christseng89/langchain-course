"""
Tool-Calling Agents with LangGraph
Building agents that can use tools
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


# Calculate Tool
@tool
def calculate(expression: str) -> str:
  """Calculate a mathematical expression. Example: calculate('2 + 2')"""
  try:
    result = eval(expression)  # Note: In production, use a safe math parser
    return f"The result of {expression} is {result}"
  except Exception as e:
    return f"Error calculating: {e}"


# Get Weather Tool
@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  # Simulated weather data
  weather_data = {
    "new york": "72°F, Sunny",
    "london": "58°F, Cloudy",
    "tokyo": "68°F, Clear",
    "paris": "65°F, Partly Cloudy",
  }
  city_lower = city.lower()
  if city_lower in weather_data:
    return f"Weather in {city}: {weather_data[city_lower]}"
  return f"Weather data not available for {city}"


# Search Web Tool
@tool
def search_web(query: str) -> str:
  """Simulate a web search for a query."""
  # Simulated search results
  search_results = {
    "python programming": "Python is a high-level programming language known for its readability and versatility.",
    "latest news": "Today's top news: AI continues to advance, impacting various industries worldwide.",
    "best restaurants in new york": "Top restaurants in New York include Le Bernardin, Per Se, and Eleven Madison Park.",
  }
  query_lower = query.lower()
  if query_lower in search_results:
    return f"Search results for '{query}': {search_results[query_lower]}"
  return f"Error: No search results found for '{query}'"


# Divide Tool
@tool
def divide(a: float, b: float) -> str:
  """Divide two numbers."""
  if b == 0:
    return "Error: Division by zero"
  result = a / b
  return f"The result of {a} divided by {b} is {result}"


# Agent State
class AgentState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


# Create Tool Agent
def create_tool_agent():
  """Create a basic tool-calling agent."""

  def agent_node(state: AgentState) -> str:
    # Generate a response using the LLM with tool access
    system_message = SystemMessage(
      content="Reply in plain text only. Do not use LaTeX or math notation "
      "(e.g. write '25 * 17' not '\\( 25 \\times 17 \\)'). When relaying a "
      "tool result, keep its wording exactly as returned."
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]])
    return {"messages": [response]}

  def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if we should continue to tools or end."""
    last_message = state["messages"][-1]
    # print(f"\033[38;5;180mLast message: \033[0m\n{last_message.content}")

    # If no tool calls, we're done
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
      return "end"
    return "tools"

  tools = [calculate, get_weather, search_web]
  llm_with_tools = LLM.bind_tools(tools)  # Bind Tools!

  # create tool node
  tool_node = ToolNode(tools)

  # create graph
  graph = StateGraph(AgentState)

  # add nodes and edges
  graph.add_node("agent", agent_node)
  graph.add_node("tools", tool_node)

  graph.add_edge(START, "agent")
  graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

  graph.add_edge("tools", "agent")  # loop back after tool execution

  return graph.compile()


# Demo Tool Agent
def demo_tool_agent():
  """Demo the tool-calling agent."""

  agent = create_tool_agent()
  save_graph_png(agent, "graphH1_tool_agent.png")

  queries = [
    "What's 25 * 17?",
    "What's the weather in Tokyo?",
    "What's 100 / 4 and what's the weather in London?",
  ]

  # print("Tool-Calling Agent Demo:\n")

  for query in queries:
    print(f"\n\033[32mQuery: {query}\033[0m")

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # Get final response
    final_message = result["messages"][-1]
    print(f"\n\033[93mResponse:\033[0m\n{final_message.content}")

    # The pattern is Human → AI(tool_calls) → ToolMessage × N → AI(final)
    print(f"Total messages: {len(result['messages'])}")


# DEMO Tool Execution Trace
def demo_tool_execution_trace():
  """Show detailed tool execution trace."""

  agent = create_tool_agent()
  save_graph_png(agent, "graphH2_tool_execution_trace.png")

  query = "Calculate 15 percent of 250 and check weather in Paris"
  result = agent.invoke({"messages": [HumanMessage(content=query)]})

  print(f"\n\033[32mQuery: {query}\033[0m")
  for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    print(f"\n\033[93m[{i + 1}] {msg_type}:\033[0m")

    if isinstance(msg, HumanMessage):
      print(f"  Content: {msg.content}")
    elif isinstance(msg, AIMessage):
      if msg.tool_calls:
        print(f"  Tool calls: {len(msg.tool_calls)}")
        for tc in msg.tool_calls:
          print(f"    - {tc['name']}({tc['args']})")
      else:
        print(f"  Content: {msg.content}")
    elif isinstance(msg, ToolMessage):
      print(f"  Tool: {msg.name}")
      print(f"  Result: {msg.content}")


# DEMO Tool with Errors
def demo_tool_with_errors():
  """Demo tool error handling."""

  tools = [divide]
  llm_with_tools = LLM.bind_tools(tools)

  def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

  def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
      return "end"
    return "tools"

  tool_node = ToolNode(tools)

  graph = StateGraph(AgentState)
  graph.add_node("agent", agent_node)
  graph.add_node("tools", tool_node)
  graph.add_edge(START, "agent")
  graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
  graph.add_edge("tools", "agent")

  agent = graph.compile()
  save_graph_png(agent, "graphH3_tool_with_errors.png")

  # print("\nTool Error Handling Demo:\n")

  queries = [
    "Divide 100 by 5",
    "Divide 100 by 0",  # Will trigger error
    "What's the weather in Taipei? In Chinese.",  # N/A
  ]

  for query in queries:
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"\n\033[32mQuery: {query}\033[0m")
    print(f"\n\033[93mResponse:\033[0m\n{result['messages'][-1].content}")
    # print("-" * 40)
    print(f"Total messages: {len(result['messages'])}")


if __name__ == "__main__":
  print_section("Demo Tool Agent")
  demo_tool_agent()

  print_section("Demo Tool Execution Trace")
  demo_tool_execution_trace()

  print_section("Demo Tool with Errors")
  demo_tool_with_errors()
