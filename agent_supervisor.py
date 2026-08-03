"""
Supervisor Architecture in LangGraph
One agent coordinates multiple specialist agents
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv()


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


SYSTEM_PROMPT = """You are a supervisor managing a team of specialists:

        1. researcher - Gathers information and facts
        2. writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far."""

RESEARCHER_PROMPT = """You are a research specialist. Gather facts and information relevant to the task. Be thorough but concise."""
WRITER_PROMPT = """You are a writing specialist. Create clear, engaging content based on the available information."""
CRITIC_PROMPT = """You are a quality critic. Review the work and provide constructive feedback. If the work is good, say so."""

PRINT_MESSAGE = False  # Set to True to print messages from each agent
PRINT_SUPERVISOR_ONLY = True  # Set to True to print only supervisor routing decisions


class SupervisorState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  next_agent: str
  task_complete: bool
  final_response: str


LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


def create_supervisor_system():
  """Create a supervisor with specialist agents."""

  # Define the routing schema
  class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
      description="The next agent to call, or FINISH if task is complete"
    )
    reasoning: str = Field(description="Why this agent was chosen")

  supervisor_llm = LLM.with_structured_output(RouteDecision)

  # Supervisor node
  def supervisor(state: SupervisorState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    decision = supervisor_llm.invoke(messages)

    if decision.next == "FINISH":
      return {"next_agent": "FINISH", "task_complete": True}

    return {
      "next_agent": decision.next,
      "messages": [
        AIMessage(content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}")
      ],
    }

  # Define specialist agents (for demo purposes, they just echo the task)
  def researcher(state: SupervisorState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", RESEARCHER_PROMPT),
        (
          "human",
          "Task context:\n{context}\n\nProvide your research findings.",
        ),
      ]
    )

    # Get task from first human message
    task = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    if PRINT_MESSAGE:
      print(f"\n\033[38;5;180m[Researcher] Task context:\033[0m\n{task}")
    response = LLM.invoke(prompt.format_messages(context=task))
    return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

  def writer(state: SupervisorState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", WRITER_PROMPT),
        ("human", "Previous work:\n{context}\n\nWrite the content."),
      ]
    )

    context = "\n".join([m.content for m in state["messages"][-5:]])
    if PRINT_MESSAGE:
      print(f"\n\033[38;5;180m[Writer] Context for writing:\033[0m\n{context}")
    response = LLM.invoke(prompt.format_messages(context=context))

    return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}

  def critic(state: SupervisorState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", CRITIC_PROMPT),
        ("human", "Work to review:\n{context}\n\nProvide your critique."),
      ]
    )

    context = "\n".join([m.content for m in state["messages"][-3:]])
    if PRINT_MESSAGE:
      print(f"\n\033[38;5;180m[Critic] Context for review:\033[0m\n{context}")
    response = LLM.invoke(prompt.format_messages(context=context))

    return {"messages": [AIMessage(content=f"[Critic] {response.content}")]}

  def finalize(state: SupervisorState) -> dict:
    # Get the last substantial response
    for msg in reversed(state["messages"]):
      if isinstance(msg, AIMessage) and "[Writer]" in msg.content:
        content = msg.content.replace("[Writer] ", "")
        return {"final_response": content}

    return {"final_response": "Task completed."}

  # Route based on supervisor decision
  def route_to_agent(state: SupervisorState) -> str:
    if state.get("task_complete"):
      return "finalize"
    return state["next_agent"]

  # 任务可能被判断为需要先查资料, 才出现 [Researcher]
  graph = StateGraph(SupervisorState)
  graph.add_node("supervisor", supervisor)
  graph.add_node("researcher", researcher)
  graph.add_node("writer", writer)
  graph.add_node("critic", critic)
  graph.add_node("finalize", finalize)

  graph.add_edge(START, "researcher")  # researcher is mandatory and always runs first

  graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
      "researcher": "researcher",
      "writer": "writer",
      "critic": "critic",
      "finalize": "finalize",
    },
  )
  # After each specialist, go back to supervisor
  graph.add_edge("researcher", "supervisor")
  graph.add_edge("writer", "supervisor")
  graph.add_edge("critic", "supervisor")
  graph.add_edge("finalize", END)

  app = graph.compile()
  save_graph_png(app, "GraphI_supervisor_graph.png")

  return app


# DEMO: Supervisor Pattern
def demo_supervisor():
  """Demo the supervisor system."""

  questions = [
    "Write a short blog post about the benefits of AI in healthcare.",
    "Write a short blog post about the benefits of AI in software development.  In Chinese",
    # "How about a short blog post about the next generation General AI with real world examples?  In Chinese",
  ]

  for question in questions:
    print(f"\033[92mQuestion: {question}\033[0m\n")
    agent = create_supervisor_system()
    result = agent.invoke(
      {
        "messages": [HumanMessage(content=question)],
        "next_agent": "",
        "task_complete": False,
        "final_response": "",
      }
    )

    print("\n\033[93mRouting Decisions:\033[0m")
    for i, msg in enumerate(result["messages"]):
      # if isinstance(msg, AIMessage) and "[Supervisor]" in msg.content:
      print(f"\n\033[38;5;180m{i + 1}). \033[0m{msg.content}")

    print(f"\n\033[93mFinal Response:\033[0m\n{result['final_response']}\n")


# DEMO: Supervisor Pattern Trace
def demo_supervisor_trace():
  """Show supervisor decision-making."""

  question = "Create a marketing tagline for a new coffee brand. In Chinese"
  print(f"\033[92mQuestion: {question}\033[0m\n")
  agent = create_supervisor_system()
  result = agent.invoke(
    {
      "messages": [HumanMessage(content=question)],
      "next_agent": "",
      "task_complete": False,
      "final_response": "",
    }
  )

  print(f"\n\033[93mRouting Decisions Trace, Total Messages: {len(result['messages'])}\033[0m")

  # Print only the supervisor routing decisions
  for i, msg in enumerate(result["messages"]):
    if PRINT_SUPERVISOR_ONLY:
      if "[Supervisor]" in msg.content:
        print(f"\n\033[38;5;180m{i + 1}). {type(msg).__name__}\033[0m:\n{msg.content}")
    else:
      print(f"\n\033[38;5;180m{i + 1}). {type(msg).__name__}\033[0m:\n{msg.content}")

  final_response = result.get("final_response", "")
  print(f"\n\033[93mFinal Response:\033[0m\n{final_response}\n")


if __name__ == "__main__":
  # print_section("Demo Supervisor")
  # demo_supervisor()

  print_section("Demo Supervisor Trace")
  demo_supervisor_trace()
