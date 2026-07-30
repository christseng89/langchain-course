"""
Checkpointing and Persistence in LangGraph
Save and resume agent state
"""

import operator
import tempfile
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
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


def to_local(iso_timestamp: str) -> str:
  """Convert a checkpoint's UTC ISO timestamp to the system's local time."""
  return datetime.fromisoformat(iso_timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# Chat State
class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], operator.add]


#                 BaseMessage
#                      ▲
#      ┌────────┬────────┬──────────┬─────────┐
#      │        │        │          │         │
# HumanMessage AIMessage SystemMessage ToolMessage

# [
#     SystemMessage("You are a banking expert."),
#     HumanMessage("Issue an LC."),
#     AIMessage("Please provide the beneficiary."),
#     ToolMessage("Customer information retrieved.")
# ]


# Demo Memory Saver
def demo_memory_saver():
  """In-memory checkpointing for development."""

  def chat(state: ChatState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

  graph = StateGraph(ChatState)
  graph.add_node("chat", chat)

  graph.add_edge(START, "chat")
  graph.add_edge("chat", END)

  app = graph.compile(checkpointer=MemorySaver())
  save_graph_png(app, "graphG1_memory_server.png")

  # Configuration with thread_id
  config = {"configurable": {"thread_id": "chat-user-123"}}
  messages = ["My name is Paulo", "What's my name?"]

  for i, message in enumerate(messages):
    print(f"\033[92mTurn {i + 1}, Query: {message}\033[0m")
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config)
    print(f"AI: {result['messages'][-1].content}\n")

  # Check full history
  state = app.get_state(config)
  print(f"\033[93mTotal messages in state: {len(state.values['messages'])}\033[0m")


# DEMO SqLite Persistence
def demo_sqlite_persistence():
  """SQLite persistence for durable storage."""

  def chat(state: ChatState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

  graph = StateGraph(ChatState)
  graph.add_node("chat", chat)

  graph.add_edge(START, "chat")
  graph.add_edge("chat", END)

  # Create temp database
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  print(f"\033[93mDatabase: {db_path}\n\033[0m")

  # First session
  with SqliteSaver.from_conn_string(db_path) as saver:  # Use SqliteSaver as saver
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "persistent-user"}}
    messages = [
      "Remember: The secret code is ALPHA-7",
      "What was the secret code?",
      "My name is Paulo",
      "What's my name?",
    ]

    for i, message in enumerate(messages):
      result = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config,
      )
      print(f"\033[92mSession {i + 1}, Message: {message}\033[0m")
      print(f"AI: {result['messages'][-1].content}\n")

    # Check full history
    state = app.get_state(config)
    print(f"\033[93mTotal messages in state: {len(state.values['messages'])}\033[0m")


# Demo Checkpoints State Inspection
def demo_state_inspection():
  """Inspect and manipulate checkpoint state."""

  def chat(state: ChatState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

  graph = StateGraph(ChatState)
  graph.add_node("chat", chat)

  graph.add_edge(START, "chat")
  graph.add_edge("chat", END)

  app = graph.compile(checkpointer=MemorySaver())
  save_graph_png(app, "graphG2_state_inspect.png")

  config = {"configurable": {"thread_id": "inspect-demo"}}

  # Build up some state
  queries = ["Hello!", "How are you?"]

  for i, query in enumerate(queries):
    print(f"\n\033[92m{i + 1}. Query: {query}\033[0m")
    app.invoke({"messages": [HumanMessage(content=query)]}, config)

    # Get current state
    state = app.get_state(config)

    print("\n\033[93mCurrent state:\033[0m")
    print(f"Next node: {state.next}")
    print(f"Message count: {len(state.values['messages'])}")

    # Get state history
    history = list(app.get_state_history(config))
    print(f"\n\033[92mState History with {len(history)} checkpoints\033[0m")
    for i, snapshot in enumerate(history):
      step = snapshot.metadata.get("step")
      source = snapshot.metadata.get("source")
      print(
        f"Checkpoint {i + 1}: {len(snapshot.values['messages'])} messages "
        f"(step={step}, \tsource={source}, \tnext={snapshot.next})"
      )

      # 出現兩個「2 messages」(Checkpoint 3 跟 Checkpoint 4) 是兩次 invoke() 呼叫的交界點


# DEMO Branching Conversations
def demo_branching_conversations():
  """Branch conversations from checkpoints."""

  def chat(state: ChatState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

  graph = StateGraph(ChatState)
  graph.add_node("chat", chat)

  graph.add_edge(START, "chat")
  graph.add_edge("chat", END)

  app = graph.compile(checkpointer=MemorySaver())
  save_graph_png(app, "graphG3_branch_conversations.png")

  # Conversations
  conversations = [
    {"chat": "Main", "thread_id": "main", "content": "What's the weather like?"},
    {
      "chat": "Branch A (Beach)",
      "thread_id": "branch-beach",
      "content": "What about a beach vacation?",
    },
    {
      "chat": "Branch B (Mountain Hiking)",
      "thread_id": "branch-mountain",
      "content": "What about mountain hiking? in Chinese.",
    },
  ]

  main_state = None
  for i, conv in enumerate(conversations):
    print(f"\033[92m\nChat: {conv['chat']}, Content: {conv['content']}\033[0m")
    config = {"configurable": {"thread_id": conv["thread_id"]}}

    if i == 0:
      # Main conversation
      result = app.invoke({"messages": [HumanMessage(content=conv["content"])]}, config)
      main_state = app.get_state(config)
    else:
      # Branch - copy Main state to this new thread, then diverge
      app.update_state(config, main_state.values)
      result = app.invoke({"messages": [HumanMessage(content=conv["content"])]}, config)

    print(f"\n\033[93mResult:\033[0m\n{result['messages'][-1].content}")


# DEMO Checkpoint Internals
def demo_checkpoint_internals():
  """
  Peek inside a checkpoint — see exactly what LangGraph saves.

  Uses a 2-node graph so we generate multiple checkpoints,
  then walks through every field in the checkpoint object.
  """

  # ── Build a 2-node graph so we get several checkpoints ──

  class TaskState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    step: str

  def analyze(state: TaskState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response], "step": "analyzed"}

  def summarize(state: TaskState) -> dict:
    summary_prompt = [
      HumanMessage(content=f"Summarize this in one sentence: {state['messages'][-1].content}")
    ]
    response = LLM.invoke(summary_prompt)
    return {"messages": [response], "step": "summarized"}

  graph = StateGraph(TaskState)
  graph.add_node("analyze", analyze)
  graph.add_node("summarize", summarize)

  graph.add_edge(START, "analyze")
  graph.add_edge("analyze", "summarize")
  graph.add_edge("summarize", END)

  app = graph.compile(checkpointer=MemorySaver())
  save_graph_png(app, "graphG4_checkpoint_internal.png")

  config = {"configurable": {"thread_id": "internals-demo"}}
  content = "Explain why the sky is blue"
  print(f"\n\033[92mQuery: {content}\033[0m")
  app.invoke(
    {"messages": [HumanMessage(content=content)], "step": ""},
    config,
  )

  # ════════════════════════════════════════════════════════
  # PART 1: What's in the CURRENT state snapshot?
  # ════════════════════════════════════════════════════════

  print("\033[93m\nPART 1: Current State Snapshot (app.get_state)\n\033[0m")

  state = app.get_state(config)

  # state.values — your actual TypedDict data
  print("\033[38;5;208m1) state.values (your state data):\033[0m")
  print(f"   step: '{state.values['step']}'")
  print(f"   messages: {len(state.values['messages'])} total\n")
  for i, msg in enumerate(state.values["messages"]):
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"\033[38;5;208m[{i + 1}]. {role}:\033[0m\n{msg.content}\n")

  # state.next — which node runs next (empty = graph finished)
  print("\033[38;5;208m2) state.next (pending node):\033[0m")
  print(f"   {state.next if state.next else '() — graph finished, no pending nodes'}")

  # state.config — the config that produced this snapshot
  print("\033[38;5;208m\n3) state.config (thread + checkpoint IDs):\033[0m")
  print(f"   thread_id:     {state.config['configurable']['thread_id']}")
  print(f"   checkpoint_id: {state.config['configurable']['checkpoint_id']}")

  # state.metadata — who created this checkpoint
  print("\033[38;5;208m\n4) state.metadata (provenance info):\033[0m")
  print(f"   source:  {state.metadata.get('source', 'N/A')}")
  print(f"   step:    {state.metadata.get('step', 'N/A')}")
  print(f"   writes:  {state.metadata.get('writes', 'N/A')}")

  # state.parent_config — pointer to the PREVIOUS checkpoint
  print("\033[38;5;208m\n5) state.parent_config (previous checkpoint):\033[0m")
  if state.parent_config:
    print(f"   parent checkpoint_id: {state.parent_config['configurable']['checkpoint_id']}")
  else:
    print("   None — this is the very first checkpoint")

  # state.created_at — timestamp
  print("\033[38;5;208m\n6) state.created_at (when saved):\033[0m")
  print(f"   {state.created_at}  (UTC)")
  print(f"   {to_local(state.created_at)}  (local)")

  # ════════════════════════════════════════════════════════
  # PART 2: Walk through ALL checkpoints (time travel)
  # ════════════════════════════════════════════════════════

  print("\033[93m\nPART 2: Full Checkpoint History (app.get_state_history)\n\033[0m")

  for i, snapshot in enumerate(app.get_state_history(config)):
    writes = snapshot.metadata.get("writes", {})
    node_name = list(writes.keys())[0] if writes else "—"

    print(f"\033[38;5;208mCheckpoint {i + 1}:\033[0m")
    print(f"  id:         {snapshot.config['configurable']['checkpoint_id'][:30]}...")
    print(f"  source:     {snapshot.metadata.get('source', '?')}")
    print(f"  step:       {snapshot.metadata.get('step', '?')}")
    print(f"  state.step: '{snapshot.values.get('step', '')}'")
    print(f"  messages:   {len(snapshot.values.get('messages', []))}")
    print(f"  next:       {snapshot.next if snapshot.next else '() — finished'}")
    print(f"  created_at: {to_local(snapshot.created_at)}  (local)")
    print(f"  written by: {node_name}")

  # ════════════════════════════════════════════════════════
  # PART 3: Jump to a specific checkpoint (rewind)
  # ════════════════════════════════════════════════════════

  print("\033[93m\nPART 3: Rewind — Jump to a Previous Checkpoint\n\033[0m")

  # Find the checkpoint right after the "analyze" node ran
  target_snapshot = None
  for snapshot in app.get_state_history(config):
    writes = snapshot.metadata.get("writes", {})
    if "analyze" in writes:
      target_snapshot = snapshot
      break

  if target_snapshot:
    target_id = target_snapshot.config["configurable"]["checkpoint_id"]
    print(f"  Found checkpoint after 'analyze' node: {target_id[:30]}...")
    print(f"  Messages at that point: {len(target_snapshot.values['messages'])}")
    print(f"  state.step at that point: '{target_snapshot.values.get('step', '')}'")

    # You can resume from this exact checkpoint
    rewind_config = {"configurable": {"thread_id": "internals-demo", "checkpoint_id": target_id}}
    rewind_state = app.get_state(rewind_config)
    print(f"\n  Loaded checkpoint — next node would be: {rewind_state.next}")
    print("  We're back to BEFORE 'summarize' ran!")
    print("  Calling invoke(None) from here would re-run 'summarize' with fresh output.")
  else:
    print("  Could not find target checkpoint.")

  # ════════════════════════════════════════════════════════
  # SUMMARY: Anatomy of a checkpoint
  # ════════════════════════════════════════════════════════

  print("\033[91m\nCHECKPOINT ANATOMY — What Gets Saved\033[0m")
  print(
    """
    state.values        → Your TypedDict data (messages, step, etc.)
    state.next          → Tuple of nodes that run next (() if done)
    state.config        → thread_id + checkpoint_id (unique address)
    state.parent_config → Previous checkpoint's address (linked list)
    state.metadata      → source, step number, which node wrote
    state.created_at    → Timestamp of when this checkpoint was saved

    \033[93mCheckpoints are saved:\033[0m
      1. BEFORE the first node runs (initial input state)
      2. AFTER each node completes (with updated state)
      3. At interrupt points (frozen state for human-in-the-loop)

    \033[93mThink of it as a linked list of snapshots:\033[0m
      [initial] --> [after analyze] --> [after summarize]
         ^               ^                    ^
       parent          parent              current (latest)
    """
  )


if __name__ == "__main__":
  print_section("Memory Saver Demo (Multi-turn conversation)")
  demo_memory_saver()

  print_section("SQLite Persistence Demo")
  demo_sqlite_persistence()

  print_section("Checkpoints State Inspection Demo")
  demo_state_inspection()

  print_section("Branching Conversations Demo")
  demo_branching_conversations()

  print_section("Checkpoint Internals Demo")
  demo_checkpoint_internals()
