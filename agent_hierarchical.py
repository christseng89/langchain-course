"""
Hierarchical Agents in LangGraph
Multi-level supervisors with department routing using subgraphs
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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


# ============================================================
# Team State Schema -> Shared state schema used across all 3 department subgraphs
# ============================================================


class TeamState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  final_answer: str


# ============================================================
# Department 1: Research Team (subgraph)
# ============================================================


def build_research_team() -> StateGraph:  # StateGraph for the research department
  """Build the research department subgraph."""

  def web_researcher(state: TeamState) -> dict:
    """Searches the web for information."""
    # Extract the query from the last human message
    query = ""
    for msg in reversed(state["messages"]):
      if isinstance(msg, HumanMessage):
        query = msg.content
        break

    content = (
      "You are a web researcher. Find key facts and data about "
      "the topic. Provide 3-4 bullet points of findings. Be specific."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        HumanMessage(content=query),
      ]
    )

    return {
      "messages": [
        AIMessage(
          content=f"[WEB RESEARCHER]: {response.content}",
          name="web_researcher",
        )
      ]
    }

  def paper_reviewer(state: TeamState) -> dict:
    """Reviews academic/technical sources."""
    query = ""
    for msg in reversed(state["messages"]):
      if isinstance(msg, HumanMessage):
        query = msg.content
        break

    content = (
      "You are an academic reviewer. Provide technical depth and "
      "cite relevant concepts or frameworks. 3-4 bullet points."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        HumanMessage(content=query),
      ]
    )

    return {
      "messages": [
        AIMessage(
          content=f"[PAPER REVIEWER]: {response.content}",
          name="paper_reviewer",
        )
      ]
    }

  def research_lead(state: TeamState) -> dict:
    """Synthesizes findings from both researchers."""

    content = (
      "You are the research lead. Synthesize the web researcher's "
      "and paper reviewer's findings into a cohesive research summary. "
      "Keep it to one to three short paragraphs."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {
      "messages": [AIMessage(content=f"[RESEARCH LEAD]: {response.content}", name="research_lead")],
      "final_answer": response.content,
    }

  # Build the research subgraph
  research_graph = StateGraph(TeamState)

  research_graph.add_node("web_researcher", web_researcher)
  research_graph.add_node("paper_reviewer", paper_reviewer)
  research_graph.add_node("research_lead", research_lead)

  # Fan-out: both researchers work in parallel
  research_graph.add_edge(START, "web_researcher")
  research_graph.add_edge(START, "paper_reviewer")

  # Fan-in: both feed into the research lead
  research_graph.add_edge("web_researcher", "research_lead")
  research_graph.add_edge("paper_reviewer", "research_lead")

  research_graph.add_edge("research_lead", END)

  return research_graph


# DEMO: Run a single department in isolation
def demo_single_department():
  """Demo a single department subgraph in isolation."""

  research_team = build_research_team().compile()
  save_graph_png(research_team, "graphL1_single_department.png")

  query = "What is retrieval-augmented generation (RAG)? Explain in simple terms. In Chinese"
  print(f"\n\033[92mQuery: {query}\033[0m\n")
  result = research_team.invoke(
    {
      "messages": [HumanMessage(content=query)],
      "final_answer": "",
    }
  )

  for msg in result["messages"]:
    if isinstance(msg, AIMessage):
      print(f"{msg.content}\n")

  print(f"\033[93mFinal Summary:\033[0m\n{result['final_answer']}")


# ============================================================
# Department 2: Content Team (subgraph)
# ============================================================


def build_content_team() -> StateGraph:  # StateGraph for the content department
  """Build the content department subgraph."""

  def content_writer(state: TeamState) -> dict:
    """Writes content based on available context."""

    content = (
      "You are a skilled content writer. Using any research or context "
      "in the conversation, write a clear, engaging short piece "
      "(one paragraph). Match a professional but accessible tone."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {"messages": [AIMessage(content=f"[WRITER]: {response.content}", name="content_writer")]}

  def content_editor(state: TeamState) -> dict:
    """Edits and polishes the writer's output."""

    content = (
      "You are a content editor. Take the writer's draft and "
      "improve clarity, fix any issues, and tighten the language. "
      "Return the polished version only."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {
      "messages": [AIMessage(content=f"[EDITOR]: {response.content}", name="content_editor")],
      "final_answer": response.content,
    }

  content_graph = StateGraph(TeamState)

  content_graph.add_node("writer", content_writer)
  content_graph.add_node("editor", content_editor)

  content_graph.add_edge(START, "writer")
  content_graph.add_edge("writer", "editor")
  content_graph.add_edge("editor", END)

  return content_graph


# ============================================================
# Department 3: Analysis Team (subgraph)
# ============================================================


def build_analysis_team() -> StateGraph:  # StateGraph for the analysis department
  """Build the analysis department subgraph."""

  def data_analyst(state: TeamState) -> dict:
    """Provides data-driven analysis."""

    content = (
      "You are a data analyst. Analyze the topic with numbers, "
      "trends, and quantitative reasoning. Provide 3-4 data-driven "
      "insights. Make up plausible stats for demonstration."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {
      "messages": [AIMessage(content=f"[DATA ANALYST]: {response.content}", name="data_analyst")]
    }

  def strategy_advisor(state: TeamState) -> dict:
    """Provides strategic recommendations."""

    content = (
      "You are a strategy advisor. Based on the data analysis in the "
      "conversation, provide 3 actionable strategic recommendations. "
      "Be specific and practical."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {
      "messages": [
        AIMessage(
          content=f"[STRATEGY ADVISOR]: {response.content}",
          name="strategy_advisor",
        )
      ],
      "final_answer": response.content,
    }

  analysis_graph = StateGraph(TeamState)

  analysis_graph.add_node("data_analyst", data_analyst)
  analysis_graph.add_node("strategy_advisor", strategy_advisor)

  analysis_graph.add_edge(START, "data_analyst")
  analysis_graph.add_edge("data_analyst", "strategy_advisor")
  analysis_graph.add_edge("strategy_advisor", END)

  return analysis_graph


# ============================================================
# Top-Level Supervisor (parent graph)
# ============================================================


def create_hierarchical_system():
  """
  Top-level supervisor that routes to department subgraphs.
  Each department is a compiled subgraph added as a single node.
  """

  # Compile department subgraphs
  research_team = build_research_team().compile()
  save_graph_png(research_team, "graphL2.1_hs_research_team.png")

  content_team = build_content_team().compile()
  save_graph_png(content_team, "graphL2.2_hs_content_team.png")

  analysis_team = build_analysis_team().compile()
  save_graph_png(analysis_team, "graphL2.3_hs_analysis_team.png")

  # Supervisor routing schema
  class DepartmentRoute(BaseModel):
    department: Literal["research", "content", "analysis"] = Field(
      description="Which department should handle this request"
    )
    reasoning: str = Field(description="Why this department was chosen")

  router_llm = LLM.with_structured_output(DepartmentRoute)

  def ceo_supervisor(state: TeamState) -> dict:
    """Top-level supervisor routes to the right department."""

    content = (
      "You are the CEO supervisor. Route the request to the right department:\n"
      "- research: Fact-finding, investigation, technical deep-dives\n"
      "- content: Writing, blog posts, marketing copy, summaries\n"
      "- analysis: Data analysis, strategy, business decisions\n\n"
      "Choose the BEST fit department."
    )
    decision = router_llm.invoke(
      [
        SystemMessage(content=content),
        *state["messages"],
      ]
    )

    return {
      "messages": [
        AIMessage(
          content=f"[CEO]: Routing to {decision.department} — {decision.reasoning}",
          name="ceo",
        )
      ]
    }

  def route_to_department(state: TeamState) -> str:
    """Read the CEO's routing decision from the last message."""
    last_ai = None
    for msg in reversed(state["messages"]):
      if isinstance(msg, AIMessage) and msg.name == "ceo":
        last_ai = msg
        break

    if last_ai and "research" in last_ai.content.lower():
      return "research_team"
    elif last_ai and "content" in last_ai.content.lower():
      return "content_team"
    elif last_ai and "analysis" in last_ai.content.lower():
      return "analysis_team"
    return "research_team"  # default

  # Build parent graph — departments are compiled subgraphs as nodes
  # 就是把 CEO 節點和三個已編譯好的部門子圖,組成一個完整的 hierarchical 系統。
  parent = StateGraph(TeamState)

  parent.add_node("ceo", ceo_supervisor)
  parent.add_node("research_team", research_team)  # compiled subgraph
  parent.add_node("content_team", content_team)  # compiled subgraph
  parent.add_node("analysis_team", analysis_team)  # compiled subgraph

  parent.add_edge(START, "ceo")
  parent.add_conditional_edges(
    "ceo",
    route_to_department,
    {
      "research_team": "research_team",
      "content_team": "content_team",
      "analysis_team": "analysis_team",
    },
  )

  parent.add_edge("research_team", END)
  parent.add_edge("content_team", END)
  parent.add_edge("analysis_team", END)

  app = parent.compile()
  save_graph_png(app, "graphL2.0_hs_ceo_route_to_department.png")
  return app


# DEMO: Show hierarchical routing with multiple queries
def demo_hierarchical_routing():
  """Demo the full hierarchical system with routing."""

  system = create_hierarchical_system()
  queries = [
    "What are the latest trends in large language models? Response in Chinese",
    "Write a short blog introduction about AI agents. Response in Chinese",
    "Should my startup invest in building AI features this year? Response in Chinese",
  ]

  for query in queries:
    print(f"\n\033[92mQuery: {query}\033[0m\n")

    result = system.invoke({"messages": [HumanMessage(content=query)], "final_answer": ""})

    # Show the CEO routing decision
    for msg in result["messages"]:
      if isinstance(msg, AIMessage) and msg.name == "ceo":
        print(f"{msg.content}")

    # Show the final answer
    print(f"\n\033[93mFinal Answer:\033[0m\n{result['final_answer']}\n")


# DEMO: Show full trace through the hierarchy
def demo_hierarchical_trace():
  """Show full trace through the hierarchy."""

  system = create_hierarchical_system()
  # print("Full Hierarchical Trace:\n")

  query = (
    "Research the impact of AI agents on software development productivity. Response in Chinese"
  )
  print(f"\n\033[92mQuery: {query}\033[0m\n")
  result = system.invoke(
    {
      "messages": [HumanMessage(content=query)],
      "final_answer": "",
    }
  )

  for i, msg in enumerate(result["messages"]):
    if isinstance(msg, AIMessage):
      # label = msg.name or "unknown"
      label = msg.name if msg.name else "unknown"
      print(f"\033[93m[Step {i}] {label}:\033[0m\n{msg.content}\n")


if __name__ == "__main__":
  print_section("Demo Single Research Department")
  demo_single_department()

  print_section("Demo Hierarchical Routing")
  demo_hierarchical_routing()

  print_section("Demo Hierarchical Trace")
  demo_hierarchical_trace()
