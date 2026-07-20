"""
Agent Handoffs in LangGraph
Passing control and context between agents
"""

import re
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


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


class HandoffState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  current_agent: str
  handoff_reason: str
  context_summary: str


class HandoffDecision(BaseModel):
  handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(
    description="Which agent to hand off to"
  )
  reason: str = Field(description="Reason for handoff")
  context: str = Field(description="Key context to pass to next agent")


def create_customer_service_system():
  def sales_agent(state: HandoffState) -> dict:
    """Sales specialist."""
    system = f"""You are a sales specialist. Context from triage: {state.get("context_summary", "None")}

            Help the customer with product questions and purchases.
            Be helpful and informative, not pushy."""

    response = LLM.invoke([SystemMessage(content=system), *state["messages"]])

    return {
      "messages": [AIMessage(content=f"[Sales] {response.content}")],
      "current_agent": "sales_complete",
    }

  def support_agent(state: HandoffState) -> dict:
    """Technical support specialist."""
    system = f"""You are a technical support specialist. Context from triage: {state.get("context_summary", "None")}

        Help the customer with technical issues.
        Be patient and provide step-by-step guidance."""

    response = LLM.invoke([SystemMessage(content=system), *state["messages"]])

    return {
      "messages": [AIMessage(content=f"[Support] {response.content}")],
      "current_agent": "support_complete",
    }

  def billing_agent(state: HandoffState) -> dict:
    """Billing specialist."""
    system = f"""You are a billing specialist. Context from triage: {state.get("context_summary", "None")}

        Help the customer with billing questions.
        Be clear about policies and next steps."""

    response = LLM.invoke([SystemMessage(content=system), *state["messages"]])

    return {
      "messages": [AIMessage(content=f"[Billing] {response.content}")],
      "current_agent": "billing_complete",
    }

  def route_from_triage(state: HandoffState) -> str:
    agent = state["current_agent"]
    if agent in ["sales", "support", "billing"]:
      return agent
    return "end"

  def triage_agent(state: HandoffState) -> dict:
    """Initial triage to route customer."""
    system = """You are a customer service triage agent. Your job is to:
        1. Understand the customer's need
        2. Route to the appropriate specialist:
           - sales: Product questions, purchases, upgrades
           - support: Technical issues, bugs, how-to questions
           - billing: Payments, invoices, refunds
           - end: Simple questions you can answer directly

        Analyze the customer's message and decide where to route them."""

    handoff_llm = LLM.with_structured_output(HandoffDecision)
    messages = [SystemMessage(content=system)] + state["messages"]
    decision = handoff_llm.invoke(messages)

    if decision.handoff_to == "end":
      # Answer directly
      response = LLM.invoke(
        [
          SystemMessage(content="Provide a brief, helpful response to the customer."),
          *state["messages"],
        ]
      )
      return {
        "messages": [AIMessage(content=f"[Triage] {response.content}")],
        "current_agent": "end",
      }

    return {
      "current_agent": decision.handoff_to,
      "handoff_reason": decision.reason,
      "context_summary": decision.context,
      "messages": [
        AIMessage(content=f"[Triage] Transferring to {decision.handoff_to}: {decision.reason}")
      ],
    }

  graph = StateGraph(HandoffState)

  graph.add_node("triage", triage_agent)
  graph.add_node("sales", sales_agent)
  graph.add_node("support", support_agent)
  graph.add_node("billing", billing_agent)

  graph.add_edge(START, "triage")
  graph.add_conditional_edges(
    "triage",
    route_from_triage,
    {"sales": "sales", "support": "support", "billing": "billing", "end": END},
  )

  graph.add_edge("sales", END)
  graph.add_edge("support", END)
  graph.add_edge("billing", END)

  return graph.compile()


# Handoff DEMO
def demo_handoffs():
  """Demo customer service handoffs."""

  print("\n\033[94mCustomer Service Handoff Demo:\033[0m")

  agent = create_customer_service_system()
  save_graph_png(agent, "graph7_handoffs.png")

  queries = [
    "My app keeps crashing when I try to upload photos",
    "I want to upgrade to the premium plan",
    "I was charged twice for my subscription",
    "I want to create new premium plan account",
    "How to unsubscribe my plan",
    "What time do you close?",
  ]

  for query in queries:
    print(f"\033[92m\nCustomer Query: {query}\033[0m")

    result = agent.invoke(
      {
        "messages": [HumanMessage(content=query)],
        "current_agent": "",
        "handoff_reason": "",
        "context_summary": "",
      }
    )

    for i, msg in enumerate(result["messages"]):
      if isinstance(msg, AIMessage):
        intro, _, rest = msg.content.partition("\n\n")
        print(f"{i}. {intro}")
        if rest:
          tag_match = re.match(r"\[(\w+)\]", intro)
          agent_name = tag_match.group(1) if tag_match else "Agent"
          print(f"\033[93m\n{agent_name} replied:\033[0m\n{rest}")

    # print("\n")


if __name__ == "__main__":
  demo_handoffs()
