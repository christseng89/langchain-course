"""
Agent Communication Patterns in LangGraph
Shared state, message passing, and blackboard pattern
"""

import json
import operator
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

CONFIDENCE_THRESHOLD = 0.96
MAX_ITERATIONS = 4
NUM_RECOMMENDATIONS = 5
NUM_DATA_POINTS_PER_ITERATION = 4


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
# Pattern 1: Message Passing
# Agents communicate through a shared message list
# ============================================================


# add_messages is used for Chat
class MessagePassingState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  current_phase: str


def create_message_passing_pipeline():
  """Agents communicate by appending messages that others can read."""

  def researcher(state: MessagePassingState) -> dict:
    """Researches the topic and posts findings as a message."""

    content = (
      "You are a researcher."
      "Read the user's question, research it, and post your findings."
      "Keep it to 2-3 sentences."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        *state["messages"],
      ]
    )
    return {
      "messages": [AIMessage(content=f"[RESEARCHER]: {response.content}", name="researcher")],
      "current_phase": "fact_checker",
    }

  def fact_checker(state: MessagePassingState) -> dict:
    """Reads the researcher's message and validates the claims."""

    content = (
      "You are a fact-checker."
      "Read the researcher's findings in the conversation and validate or challenge them."
      "Keep it to 2-3 sentences."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        *state["messages"],
      ]
    )
    return {
      "messages": [AIMessage(content=f"[FACT-CHECKER]: {response.content}", name="fact_checker")],
      "current_phase": "summarizer",
    }

  def summarizer(state: MessagePassingState) -> dict:
    """Reads all previous messages and creates a final summary."""

    content = (
      "You are a summarizer."
      "Read the researcher's findings and the fact-checker's review."
      "Produce a final, accurate summary. Keep it to 2-3 sentences."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        *state["messages"],
      ]
    )
    return {
      "messages": [AIMessage(content=f"[SUMMARY]: {response.content}", name="summarizer")],
      "current_phase": "done",
    }

  graph = StateGraph(MessagePassingState)

  graph.add_node("researcher", researcher)
  graph.add_node("fact_checker", fact_checker)
  graph.add_node("summarizer", summarizer)

  graph.add_edge(START, "researcher")
  graph.add_edge("researcher", "fact_checker")
  graph.add_edge("fact_checker", "summarizer")
  graph.add_edge("summarizer", END)

  return graph.compile()


# Message passing demo
def demo_message_passing():
  """Demo message passing between agents."""
  agent = create_message_passing_pipeline()
  save_graph_png(agent, "graphK1_message_passing.png")

  query = "What are the main benefits of renewable energy? In Chinese."
  print(f"\n\033[32mUser query: {query}\033[0m")

  result = agent.invoke(
    {
      "messages": [HumanMessage(content=query)],
      "current_phase": "researcher",
    }
  )

  print(f"\n\033[33mFinal Summary with Number of messages: {len(result['messages'])}\033[0m\n")
  for msg in result["messages"]:
    if isinstance(msg, AIMessage):
      print(f"{msg.content}\n")


# ============================================================
# Pattern 2: Shared State (Typed Fields)
# Agents communicate through structured state fields
# operator.add is used to combine lists from multiple agents
# ============================================================


class SharedFieldsState(TypedDict):
  query: str
  # Each agent writes to its own field — others can read it
  raw_data: Annotated[list[dict], operator.add]
  analysis: str
  recommendations: list[str]
  confidence_score: float
  iteration: int
  gaps: str


def create_shared_fields_pipeline():
  """Agents communicate through typed state fields, not messages."""

  def data_collector(state: SharedFieldsState) -> dict:
    """Collects data and writes to the raw_data field."""

    content = (
      f"You are a data collector. Given the query, produce {NUM_DATA_POINTS_PER_ITERATION} NEW data points "
      "as a JSON array of objects with 'source' and 'finding' keys. "
      "Do not repeat any finding already collected. If gaps are pointed out "
      "by the analyst, prioritize filling those specific gaps; otherwise dig "
      "into a different angle, source, or level of detail. "
      "Return ONLY the JSON array, no markdown."
    )

    existing_findings = [d.get("finding", "") for d in state.get("raw_data", [])]
    human_content = state["query"]
    if existing_findings:
      already_collected = "\n".join(f"- {f}" for f in existing_findings)
      human_content += f"\n\nAlready collected (do not repeat these):\n{already_collected}"
    if state.get("gaps"):
      human_content += f"\n\nGaps identified by the analyst — focus here:\n{state['gaps']}"

    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        HumanMessage(content=human_content),
      ]
    )

    try:
      data = json.loads(response.content)
    except json.JSONDecodeError:
      # If the LLM response is not valid JSON, wrap it in a single data point
      data = [{"source": "llm", "finding": response.content}]

    iteration = state.get("iteration", 0) + 1
    print(f"\033[38;5;180m[data_collector] iteration {iteration}\033[0m")
    return {"raw_data": data, "iteration": iteration}

  def analyst(state: SharedFieldsState) -> dict:
    """Reads raw_data field, writes analysis and confidence."""
    data_summary = json.dumps(state["raw_data"], indent=2)

    content = (
      "You are a data analyst. Analyze the collected data and provide: "
      "1) A brief analysis (2-3 sentences), "
      "2) A confidence score from 0.0 to 1.0, reflecting how well the "
      "evidence supports a solid answer. More independent, non-redundant "
      "data points covering different angles should raise your confidence; "
      "thin or repetitive evidence should keep it low, and "
      f"3) If confidence is below {CONFIDENCE_THRESHOLD}, specific gaps the data collector "
      "should fill next (missing angles, unanswered sub-questions, weak "
      f"sourcing). If confidence is already {CONFIDENCE_THRESHOLD}+, leave this empty. "
      "Format: ANALYSIS: <text>\nCONFIDENCE: <number>\nGAPS: <text or empty>"
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        HumanMessage(content=f"Query: {state['query']}\n\nData:\n{data_summary}"),
      ]
    )

    content = response.content
    analysis = content
    confidence = 0.7  # default
    gaps = ""

    if "CONFIDENCE:" in content:
      analysis_part, _, rest = content.partition("CONFIDENCE:")
      analysis = analysis_part.replace("ANALYSIS:", "").strip()
      confidence_part, _, gaps_part = rest.partition("GAPS:")
      try:
        confidence = float(confidence_part.strip())
        print(f"\033[38;5;180mParsed confidence score: {confidence}\033[0m")
      except ValueError:
        confidence = 0.7
      gaps = gaps_part.strip()

    return {"analysis": analysis, "confidence_score": confidence, "gaps": gaps}

  def advisor(state: SharedFieldsState) -> dict:
    """Reads analysis + confidence, writes recommendations."""

    content = (
      "You are a strategic advisor. Based on the analysis and confidence score, "
      f"provide {NUM_RECOMMENDATIONS} actionable recommendations. Return them as a JSON array of strings. "
      "Return ONLY the JSON array, no markdown."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        HumanMessage(
          content=(
            f"Query: {state['query']}\n"
            f"Analysis: {state['analysis']}\n"
            f"Confidence: {state['confidence_score']}"
          )
        ),
      ]
    )

    try:
      recs = json.loads(response.content)
    except json.JSONDecodeError:
      recs = [response.content]

    return {"recommendations": recs}

  def should_retry_node(_state: SharedFieldsState) -> dict:
    """Pass-through node so the retry/advance decision shows up in the graph diagram."""
    return {}

  def should_retry(state: SharedFieldsState) -> Literal["data_collector", "advisor"]:
    """Loop back for more data until confidence is high enough or we've tried enough times."""
    if state["confidence_score"] < CONFIDENCE_THRESHOLD and state["iteration"] < MAX_ITERATIONS:
      return "data_collector"
    return "advisor"

  graph = StateGraph(SharedFieldsState)

  graph.add_node("data_collector", data_collector)
  graph.add_node("analyst", analyst)
  graph.add_node("should_retry", should_retry_node)
  graph.add_node("advisor", advisor)

  graph.add_edge(START, "data_collector")
  graph.add_edge("data_collector", "analyst")
  graph.add_edge("analyst", "should_retry")
  graph.add_conditional_edges(
    "should_retry", should_retry, {"data_collector": "data_collector", "advisor": "advisor"}
  )
  graph.add_edge("advisor", END)

  return graph.compile()


# Demo shared state fields between agents
def demo_shared_state():
  """Demo shared state fields between agents."""
  agent = create_shared_fields_pipeline()
  save_graph_png(agent, "graphK2_shared_state.png")

  queires = [
    # "Should a small business invest in AI automation in 2026? In Chinese.",
    # "一個開發銀行軟件公司,軟件開發過程中使用與未使用AI的主要差別是什麼? 請提供一些實際的例子。 In Chinese.",
    # "What is the benefit to use IDP (internal developer platform) for a software development company? In Chinese.",
    # "銀行使用AI提升工作效率的作業有哪些 in 2026? 請提供一些實際的案例。 In Chinese.",
    "銀行使用AI提升Trade Finance工作效率的作業有哪些 in 2026? 請提供一些實際的案例。 In Chinese.",
  ]

  for query in queires:
    print(f"\n\033[32mUser query: {query}\033[0m")

    result = agent.invoke(
      {
        "query": query,
        "raw_data": [],
        "analysis": "",
        "recommendations": [],
        "confidence_score": 0.0,
        "iteration": 0,
        "gaps": "",
      }
    )

    print(f"\n\033[93mData collected: {len(result['raw_data'])} points\033[0m")
    for i, d in enumerate(result["raw_data"], 1):
      print(f"{i}. [{d.get('source', 'N/A')}] {d.get('finding', 'N/A')}")

    print(f"\nAnalysis: {result['analysis']}")
    print(f"Confidence: {result['confidence_score']}")
    print(f"Iterations: {result['iteration']}")

    print("\n\033[93mRecommendations:\033[0m")
    for i, rec in enumerate(result["recommendations"], 1):
      print(f"{i}. {rec}")


# ============================================================
# Pattern 3: Blackboard Pattern
# Combines shared workspace + messages + iterative refinement
# ============================================================


class BlackboardState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  # Blackboard fields — the shared workspace
  topic: str
  drafts: Annotated[list[str], operator.add]
  critiques: Annotated[list[str], operator.add]
  iteration: int
  is_approved: bool


def create_blackboard_system():
  """
  Blackboard pattern: multiple agents read/write a shared workspace.
  A drafter writes, a critic reviews, and they iterate until approved.
  """

  class ApprovalDecision(BaseModel):
    approved: bool = Field(description="Whether the draft is good enough")
    feedback: str = Field(description="Specific feedback if not approved")

  critic_llm = LLM.with_structured_output(ApprovalDecision)

  def drafter(state: BlackboardState) -> dict:
    """Reads critiques from blackboard, writes improved draft."""
    context_parts = [f"Topic: {state['topic']}"]

    if state["drafts"]:
      context_parts.append(f"Previous draft: {state['drafts'][-1]}")
    if state["critiques"]:
      context_parts.append(f"Feedback to address: {state['critiques'][-1]}")

    context = "\n".join(context_parts)

    content = (
      "You are a skilled writer. Write or revise a short paragraph "
      "(3-4 sentences) based on the topic and any feedback provided. "
      "If there's feedback, directly address it in your revision."
    )
    response = LLM.invoke(
      [
        SystemMessage(content=(content)),
        HumanMessage(content=context),
      ]
    )

    print(f"\033[38;5;180m[drafter] iteration {state['iteration'] + 1}\033[0m")
    return {
      "drafts": [response.content],
      "messages": [
        AIMessage(
          content=f"[DRAFTER iteration {state['iteration'] + 1}]: {response.content}",
          name="drafter",
        )
      ],
      "iteration": state["iteration"] + 1,
    }

  def critic(state: BlackboardState) -> dict:
    """Reads latest draft from blackboard, writes critique or approves."""
    latest_draft = state["drafts"][-1] if state["drafts"] else "No draft yet"

    content = (
      "You are a strict, hard-to-please editor. Reject generic, vague, or "
      "buzzword-heavy writing — phrases like 'improves efficiency' or "
      "'accelerates development' with no concrete example, number, or "
      "real-world specific do NOT count as good enough. "
      "Approve ONLY if the draft includes at least one concrete, specific "
      "example, detail, or piece of evidence per claim it makes. "
      "By default, assume the first draft has room to improve and require "
      "at least one round of revision — only approve on iteration 1 if the "
      "draft is truly exceptional with zero notable weaknesses. "
      "When you reject, the feedback MUST name the specific claim that "
      "needs a concrete example or detail added. "
      f"If iteration is {MAX_ITERATIONS} or more, be more lenient."
    )
    decision = critic_llm.invoke(
      [
        SystemMessage(content=(content)),
        HumanMessage(
          content=(
            f"Topic: {state['topic']}\nIteration: {state['iteration']}\nDraft: {latest_draft}"
          )
        ),
      ]
    )

    # Force approval after MAX_ITERATIONS to prevent infinite loops
    print(f"\033[38;5;180m[critic] Approved: {decision.approved}\033[0m")
    approved = decision.approved or state["iteration"] >= MAX_ITERATIONS

    content = f"[CRITIC]: {'APPROVED' if approved else 'REVISION NEEDED'} - {decision.feedback}"
    result = {
      "is_approved": approved,
      "messages": [
        AIMessage(
          content=content,
          name="critic",
        )
      ],
    }

    if not approved:
      result["critiques"] = [decision.feedback]

    return result

  def route_after_critic_node(_state: BlackboardState) -> dict:
    """Pass-through node so the retry/approve decision shows up in the graph diagram."""
    return {}

  def route_after_critic(state: BlackboardState) -> Literal["drafter", "end"]:
    """Loop back to drafter if not approved."""
    if state["is_approved"]:
      return "end"
    return "drafter"

  graph = StateGraph(BlackboardState)

  graph.add_node("drafter", drafter)
  graph.add_node("critic", critic)
  graph.add_node("route_after_critic", route_after_critic_node)

  graph.add_edge(START, "drafter")
  graph.add_edge("drafter", "critic")
  graph.add_edge("critic", "route_after_critic")
  graph.add_conditional_edges(
    "route_after_critic", route_after_critic, {"drafter": "drafter", "end": END}
  )

  return graph.compile()


def demo_blackboard():
  """Demo blackboard iterative refinement."""
  agent = create_blackboard_system()
  save_graph_png(agent, "graphK3_blackboard.png")

  queries = [
    "Why is LangGraph great for building multi-agent systems? Give some real world examples. In Chinese.",
    "What are the main benefits to use AI for software development life cycle? In Chinese.",
  ]

  for query in queries:
    # query = "Why is LangGraph great for building multi-agent systems? Give some real world examples. In Chinese."
    print(f"\n\033[32mUser query: {query}\033[0m")
    result = agent.invoke(
      {
        "messages": [],
        "topic": query,
        "drafts": [],
        "critiques": [],
        "iteration": 0,
        "is_approved": False,
      }
    )

    print(f"\n\033[93mTotal Iterations: {result['iteration']}\033[0m")
    print(f"Approved: {result['is_approved']}")

    print("\n\033[93mConversation:\033[0m")
    for msg in result["messages"]:
      if isinstance(msg, AIMessage):
        print(f"{msg.content}\n")

    print(f"\033[93mFinal Draft:\033[0m\n{result['drafts'][-1]}")


if __name__ == "__main__":
  # print_section("Demo Message Passing - add_messages")
  # demo_message_passing()

  # print_section("Demo Shared State - operator.add")
  # demo_shared_state()

  print_section("Demo Blackboard - add_messages + embedded operator.add")
  demo_blackboard()
