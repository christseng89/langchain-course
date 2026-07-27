"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify, and resume
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


# ─── Helper for visual separation ───
def phase_banner(phase_num: int, title: str):
  print(f"\033[92m\nPHASE {phase_num}: {title}\033[0m")


def step_print(icon: str, label: str, detail: str = "", color: str = "\033[38;5;208m"):
  print(f"{color}\n{icon}  [{label}] {detail}\033[0m")


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


# ════════════════════════════════════════════════════════
# DEMO 1: Interrupt for Approval
# ════════════════════════════════════════════════════════


class ApprovalState(TypedDict):
  request: str
  draft: str
  approved: bool
  feedback: str
  final: str


# Interrupt for Approval Demo
def demo_interrupt_for_approval():
  """Interrupt execution for human approval."""

  def create_draft(state: ApprovalState) -> dict:
    step_print("📝", "DRAFT NODE", "Entering create_draft node...")
    print(f'   Request: "{state["request"]}"')

    response = LLM.invoke(f"Create a professional response for: {state['request']}")

    print(f"   Draft Generated {len(response.content)} words")
    print(f"   Draft Preview: {response.content[:20]}...")
    return {"draft": response.content}

  def wait_for_approval(state: ApprovalState) -> dict:
    step_print("👁️", "APPROVAL NODE", "Entering wait_for_approval node...")
    print(f"\033[95m   Approved: {state['approved']}\033[0m")
    print(f"   Feedback: '{state['feedback']}'" if state["feedback"] else "   Feedback: (none yet)")
    # This node is where we'll interrupt
    return state

  def check_feedback(state: ApprovalState) -> dict:
    step_print("🔀", "CHECK FEEDBACK NODE", "Deciding whether to finalize or wait for real input")
    return {}

  def route_after_check(state: ApprovalState) -> Literal["finalize", "approval"]:
    if state["approved"] or state["feedback"]:
      return "finalize"
    print("\033[91m   No feedback provided yet -> looping back to wait for real input\033[0m")
    return "approval"

  def finalize(state: ApprovalState) -> dict:
    step_print("📦", "FINALIZE NODE", "Entering finalize node")
    print(f"\033[95m   Approved: {state['approved']}\033[0m")

    if state["approved"]:
      print("\033[95m   Action: Using draft as-is (human approved)\033[0m")
      return {"final": state["draft"]}
    else:
      print("\033[95m   Action: Revising draft based on feedback...\033[0m")
      print(f'   Feedback: "{state["feedback"]}"')
      # Incorporate feedback
      response = LLM.invoke(
        f"Revise this draft based on feedback:\n\n"
        f"Draft: {state['draft']}\n\n"
        f"Feedback: {state['feedback']}"
      )

      approved_status = True
      print(f"   Revised draft generated ({len(response.content)} words)")
      print(f"\033[95m   Approved updated to: {approved_status}\033[0m")
      return {"final": response.content, "approved": approved_status}

  graph = StateGraph(ApprovalState)
  graph.add_node("draft", create_draft)
  graph.add_node("approval", wait_for_approval)
  graph.add_node("finalize", finalize)
  graph.add_node("check_feedback", check_feedback)

  graph.add_edge(START, "draft")
  graph.add_edge("draft", "approval")
  graph.add_edge("approval", "check_feedback")
  graph.add_conditional_edges(
    "check_feedback", route_after_check, {"finalize": "finalize", "approval": "approval"}
  )
  graph.add_edge("finalize", END)

  # Compile with checkpointer and interrupt
  memory = MemorySaver()
  app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approval"],  # Pause before this node
  )

  save_graph_png(app, "graphE_interrupt_approval.png")

  # ─── PHASE 1: Run until interrupt ───
  phase_banner(1, "RUN UNTIL INTERRUPT")

  config = {"configurable": {"thread_id": "demo-1"}}
  request = "Write a thank-you email for a job interview"
  result = app.invoke(
    {
      "request": request,
      "draft": "",
      "approved": False,
      "feedback": "",
      "final": "",
    },
    config,
  )

  step_print("⏸️", "PAUSED", "Graph hit interrupt_before='approval'")
  print(f"   Request: {request}")
  print(f"   Draft Preview: {result['draft'][:20]}...")
  print(f"\033[95m   Final is empty: '{result['final']}'\033[0m")

  current_state = app.get_state(config)
  step_print("🔍", "INSPECT APP CURRENT STATE", "Inspect current status BEFORE feedback")
  print(f"   Next node(s): {current_state.next}")
  print(f"   State keys: {list(current_state.values.keys())}")
  print(f"   Draft filled: {'Yes' if current_state.values['draft'] else 'No'}")
  print(f"   Final filled: {'Yes' if current_state.values['final'] else 'No'}")
  print(f"   Draft words: {len(current_state.values['draft'])}")
  print(f"   Final words: {len(current_state.values['final'])}")
  print(f"\033[95m   Approved: {current_state.values['approved']}\033[0m")

  # ─── PHASE 2: Human provides feedback and resume ───
  phase_banner(2, "HUMAN INJECTS FEEDBACK + RESUME -> FINAL APPROVE")

  # Blocks here for real terminal input -- the script itself now waits,
  # not just the graph. Blank input keeps re-prompting instead of silently
  # counting as approval; type 'approve' to accept the draft as-is.

  # feedback = "Make it more concise and add specific mention of the company culture"

  feedback = ""

  while not feedback:
    feedback = input('   Enter feedback, or type "approve" to accept as-is: ').strip()

  approved = feedback.lower() == "approve"
  if approved:
    feedback = ""
    print("   Human approved the draft as-is")
  else:
    print(f'   Human feedback: "{feedback}"')

  # Update state with human input
  app.update_state(
    config,
    {"approved": approved, "feedback": feedback},
  )
  final_result = app.invoke(None, config)

  current_state = app.get_state(config)
  step_print("🔍", "INSPECT APP CURRENT STATE", "Inspect current status AFTER feedback")
  print(f"\033[95m   Approved: {current_state.values['approved']}\033[0m")
  print(f"   Next node(s): {current_state.next}")
  print(f"   Draft words: {len(current_state.values['draft'])}")
  print(f"   Final words: {len(current_state.values['final'])}")

  step_print("✅", "WORKFLOW COMPLETE", "", color="\033[92m")
  print(f"\nFinal result with ({len(final_result['final'])} words):")
  print(f"\033[93mFinal Preview:\033[0m\n\n{final_result['final']}")


# ════════════════════════════════════════════════════════
# DEMO 2: Iterative Review (Human-in-the-Loop + Cycles)
# ════════════════════════════════════════════════════════


class ReviewState(TypedDict):
  document: str
  review_comments: list[str]
  revision_count: int
  status: str


# Iterative Review DEMO
def demo_iterative_review():
  """Multiple rounds of human review."""

  def submit_for_review(state: ReviewState) -> dict:
    status = "pending_review"
    step_print("📋", "SUBMIT NODE", f"Round {state['revision_count'] + 1}")
    print(f"\033[95m   Status incoming: '{state['status']}'\033[0m")
    print(f"   Setting status to '{status}'")
    print(f"   Document preview: {state['document'][:80]}...")
    return {"status": status}

  def apply_feedback(state: ReviewState) -> dict:
    step_print("🔧", "APPLY FEEDBACK NODE", f"Revision #{state['revision_count'] + 1}")

    if not state["review_comments"]:
      print("\033[95m   No comments to apply. Passing through.\033[0m")
      return state

    feedback = state["review_comments"][-1]
    print(f'\033[95m   Feedback to apply: "{feedback}"\033[0m')
    print(f"   Current document `{len(state['document'])}` words.")
    print(f"   Current document Preview: {state['document'][:60]}...")

    response = LLM.invoke(
      f"Revise this document based on feedback:\n\n"
      f"Document: {state['document']}\n\n"
      f"Feedback: {feedback}"
    )

    print(f"\n   Revised document `{len(response.content)}` words.")
    print(f"   Revised document Preview: {response.content[:60]}...")

    return {
      "document": response.content,
      "revision_count": state["revision_count"] + 1,
      "status": "revised",
    }

  def route_after_review(state: ReviewState) -> Literal["apply", "approved"]:
    step_print("🔀", "ROUTER", f"Checking status: '{state['status']}'")
    if state["status"] == "approved":
      print("\033[38;5;130mDecision: APPROVED\033[0m -> routing to 'done' node")
      return "approved"
    print("\033[38;5;130mDecision: NOT APPROVED\033[0m -> routing to 'apply' node")
    return "apply"

  def finalize(state: ReviewState) -> dict:
    step_print("🏁", "DONE NODE", "Finalizing document")
    print(f"   Total revisions: {state['revision_count']}")
    print(f"   Final document: {state['document'][:100]}...")
    return {"status": "finalized"}

  graph = StateGraph(ReviewState)

  graph.add_node("submit", submit_for_review)
  graph.add_node("apply", apply_feedback)
  graph.add_node("approved", finalize)

  graph.add_conditional_edges(
    "submit", route_after_review, {"apply": "apply", "approved": "approved"}
  )

  graph.add_edge(START, "submit")
  graph.add_edge("apply", "submit")  # Loop for more reviews
  graph.add_edge("approved", END)

  app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["submit"])
  save_graph_png(app, "graphF_interative_review.png")

  # ─── ROUND 0: Initial submission ───
  document = "AI is technology that helps computers think."
  phase_banner(0, f"INITIAL SUBMISSION - {document}")

  config = {"configurable": {"thread_id": "review-1"}}
  result = app.invoke(
    {
      "document": document,
      "review_comments": [],
      "revision_count": 0,
      "status": "",
    },
    config,
  )
  current_state = app.get_state(config)

  step_print("⏸️", "PAUSED", "Graph hit interrupt_before='submit'")
  print(f"\033[95m   Current revision: {current_state.values['revision_count']}\033[0m")
  print(f"   Next node: {current_state.next}")

  # ─── ROUND 1: Reviewer wants changes ───
  description1 = "HUMAN REVIEWER REQUESTS 1ST CHANGES"
  phase_banner(1, description1)

  status = "needs_revision"
  feedback_1 = "Add more technical depth and real-world examples"
  print(f'   Reviewer says: "{feedback_1}", set status: {status}')
  app.update_state(config, {"review_comments": [feedback_1], "status": status})
  result = app.invoke(None, config)
  current_state = app.get_state(config)

  step_print("⏸️", "PAUSED", description1)
  print(f"\033[95m   Revisions: {result['revision_count']}\033[0m")
  print(f"   Revised document Preview: {result['document'][:60]}...")
  print(f"   Next node: {current_state.next}")

  # ─── ROUND 2: Reviewer wants more changes ───
  description2 = "HUMAN REVIEWER REQUESTS MORE CHANGES"
  phase_banner(2, description2)

  feedback_2 = "Good improvement! Now add a concrete example of neural networks"
  print(f'   Reviewer says: "{feedback_2}", set status: {status}')
  app.update_state(config, {"review_comments": [feedback_2], "status": status})
  result = app.invoke(None, config)

  step_print("⏸️", "PAUSED", description2)
  print(f"\033[95m   Revisions: {result['revision_count']}\033[0m")
  print(f"   Revised document Preview: {result['document'][:60]}...")

  # ─── ROUND 3: Reviewer approves ───
  phase_banner(3, "REVIEWER APPROVES")

  status = "approved"
  print(f"\033[38;5;130m   Reviewer sets status: {status}\033[0m")
  app.update_state(config, {"status": status})
  final = app.invoke(None, config)

  # ─── FINAL SUMMARY ───
  step_print("✅", "WORKFLOW COMPLETE", color="\033[92m")
  print(f"\n\033[95mTotal revisions: {final['revision_count']}\033[0m")
  print(f"Final status: {final['status']}")
  print(f"Final document {len(final['document'])} words")

  print(f"\033[93mFinal document:\033[0m\n\n{final['document']}")


if __name__ == "__main__":
  print_section("Demo 1: Interrupt for Approval")
  demo_interrupt_for_approval()

  print_section("Demo 2: Iterative Review")
  demo_iterative_review()
