"""
LangGraph Core Concepts
StateGraph, nodes, edges, and basic patterns
"""

import operator
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

# LangGraph
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

load_dotenv()

LLM = init_chat_model("gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m\n")


# Simple Graph Demo
# Simple State
class SimpleState(TypedDict):
  input: str
  output: str
  step: int


def demo_simple_graph():
  # define node functions
  def process(state: SimpleState) -> dict:
    # simple processing logic, for demo purposes
    return {"output": state["input"].upper(), "step": state["step"] + 1}

  # Create Graph
  graph = StateGraph(SimpleState)

  # Add Nodes
  graph.add_node("Process", process)

  # Add Edges
  graph.add_edge(START, "Process")
  graph.add_edge("Process", END)

  # execute graph/ compile
  app = graph.compile()

  # # visualize the graph
  # print("\n\033[93m--- Mermaid Graph ---\033[0m")
  # print(app.get_graph().draw_mermaid())

  # save as PNG
  save_graph_png(app, "graph1_simple.png")

  result = app.invoke({"input": "hello", "output": "", "step": 0})
  print("\033[92mSimple Graph Result:\033[0m")
  print(f"Input: {result['input']}, Output: {result['output']}, Step: {result['step']}")


# Accumulating State DEMO
class AccumulatingState(TypedDict):
  messages: Annotated[list[str], operator.add]  # lists concatenate when merged
  count: Annotated[int, operator.add]  # counts sum when merged


def demo_accumulating_state():
  def step_one(state: AccumulatingState) -> dict:
    return {"messages": ["Step 1 executed"], "count": 1}

  def step_two(state: AccumulatingState) -> dict:
    return {"messages": ["Step 2 executed"], "count": 2}

  # Create Graph, Add Nodes, Add Edges then Compile Graph
  graph = StateGraph(AccumulatingState)
  graph.add_node("Step_one", step_one)
  graph.add_node("Step_two", step_two)

  graph.add_edge(START, "Step_one")
  graph.add_edge("Step_one", "Step_two")
  graph.add_edge("Step_two", END)

  app = graph.compile()

  # # visualize the graph
  # print("\n\033[93m--- Mermaid Graph ---\033[0m")
  # print(app.get_graph().draw_mermaid())

  # save as PNG
  save_graph_png(app, "graph2_accumulating.png")

  result = app.invoke({"messages": [], "count": 0})
  print("\n\033[92mAccumulating State Result:\033[0m")
  print(f"Messages: {result['messages']}")
  print(f"Count: {result['count']}")  # 1 + 2 = 3


# === Message State (Common Pattern) ===


# DEMO Message State w LLM
class MessageState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


def demo_message_state():
  def chat_node(state: MessageState) -> dict:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

  # Create Graph, Add Nodes, Add Edges then Compile Graph
  graph = StateGraph(MessageState)
  graph.add_node("Chat_node", chat_node)
  graph.add_edge(START, "Chat_node")
  graph.add_edge("Chat_node", END)

  app = graph.compile()
  # save as PNG
  save_graph_png(app, "graph3_message_state.png")

  result = app.invoke({"messages": [HumanMessage(content="Say Hello in Chinese")]})
  print("\033[92mMessage State Result:\033[0m")
  for msg in result["messages"]:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"{role}: {msg.content}")


# Multi-Node Graph w LLM
class MultiStepState(TypedDict):
  input: str
  analyzed: str
  enhanced: str
  final: str


def demo_multi_node_graph():
  UNKNOWN = "I DON'T KNOW"

  def is_unknown(text: str) -> bool:
    # The model doesn't always reproduce the literal string exactly (e.g. adds
    # trailing punctuation or changes case), so normalize before comparing.
    return text.strip().rstrip(".!?").strip().upper() == UNKNOWN

  def analyze_node(state: MultiStepState) -> dict:
    response = LLM.invoke(
      [
        HumanMessage(
          content=f"Analyze the following input and summarize it in one sentence: {state['input']}. "
          f"If you don't know, just say '{UNKNOWN}'"
        )
      ]
    )
    return {"analyzed": response.content}

  def enhance(state: MultiStepState) -> dict:
    # Short-circuit in code instead of asking the LLM to notice its own earlier
    # refusal -- relying on the model to compare a value against a literal
    # string inside a prose instruction is unreliable, and doubles the cost
    # of a call that has nothing to enhance anyway.
    if is_unknown(state["analyzed"]):
      return {"enhanced": UNKNOWN}

    response = LLM.invoke(
      [
        HumanMessage(
          content=f"Take the following analysis and enhance it with more details: {state['analyzed']}"
        )
      ]
    )
    return {"enhanced": response.content}

  def finalize(state: MultiStepState) -> dict:
    if is_unknown(state["enhanced"]):
      return {"final": UNKNOWN}

    response = LLM.invoke(
      [
        HumanMessage(
          content=f"Take the following enhanced analysis and finalize it into a concise summary: {state['enhanced']}"
        )
      ]
    )
    return {"final": response.content}

  # Create Graph, Add Nodes, Add Edges then Compile Graph
  graph = StateGraph(MultiStepState)
  graph.add_node("Analyze_node", analyze_node)
  graph.add_node("Enhance_node", enhance)
  graph.add_node("Finalize_node", finalize)

  graph.add_edge(START, "Analyze_node")
  graph.add_edge("Analyze_node", "Enhance_node")
  graph.add_edge("Enhance_node", "Finalize_node")
  graph.add_edge("Finalize_node", END)

  app = graph.compile()

  # # # visualize the graph
  # print("\n--- Mermaid Graph ---")
  # print(app.get_graph().draw_mermaid())

  # save as PNG
  save_graph_png(app, "graph4_multi_nodes.png")

  questions = ["Artificial intelligence", "Agentic AI", "LangChain", "LangGraph", "Claude Code"]

  for query in questions:
    print(f"\033[92mMulti-Node Graph Result: {query}\033[0m")
    result = app.invoke({"input": query})
    # print(f"Input: {result['input']}") # input = query
    print(f"\nAnalyzed:\n\n{result['analyzed']}")
    print(f"\nEnhanced Preview:\n\n{result['enhanced'][:200]}...")
    if result["final"] == UNKNOWN:
      print(f"\n\033[91mFinal: {result['final']}\033[0m\n")
    else:
      print(f"\n\033[38;5;208mFinal:\n\n{result['final']}\033[0m\n")


# Exercise first LangGraph w LLM
def exercise_first_langgraph():
  """
  EXERCISE: Create a LangGraph that:
  1. Takes a topic as input
  2. Node 1: Generates 3 questions about the topic
  3. Node 2: Answers one of the questions
  4. Returns both questions and answer
  """

  class QAState(TypedDict):
    topic: str
    questions: str
    answer: str

  # llm = init_chat_model("gpt-4o-mini", temperature=0)

  def generate_questions(state: QAState) -> dict:
    response = LLM.invoke(
      f"Generate 3 interesting questions about: {state['topic']}\nFormat: numbered list"
    )
    return {"questions": response.content}

  def answer_1st_question(state: QAState) -> dict:
    response = LLM.invoke(f"Answer the first question from this list:\n{state['questions']}")
    return {"answer": response.content}

  # Create Graph, Add Nodes, Add Edges then Compile Graph
  graph = StateGraph(QAState)

  graph.add_node("Generate_questions", generate_questions)
  graph.add_node("Answer_1st_question", answer_1st_question)

  graph.add_edge(START, "Generate_questions")
  graph.add_edge("Generate_questions", "Answer_1st_question")
  graph.add_edge("Answer_1st_question", END)

  app = graph.compile()

  # Save as PNG
  save_graph_png(app, "graph5_first_excercise.png")

  queries = ["The future of renewable energy", "The future of AI application, in Chinese"]
  for query in queries:
    result = app.invoke({"topic": query})

    print(f"\033[92mExercise Query: {query}\033[0m")
    # print(f"  Topic: {result['topic']}")
    print(f"\n\033[93mGenerated Questions:\n{result['questions']}\033[0m")
    print(f"\n\033[38;5;208mAnswer 1st Question:\n{result['answer']}\033[0m\n")


# Exercise second LangGraph w LLM (Map-Reduce w/ Send)
def exercise_second_langgraph():
  """
  EXERCISE: Extend the first exercise so it answers ALL generated questions,
  not just the first one:
  1. Takes a topic as input
  2. Node 1: Generates 3 questions about the topic
  3. Fan-out: dispatch each question to its own "Answer_question" node in parallel,
     using the Send API since the number of questions isn't known until runtime
  4. Fan-in: an accumulating reducer collects every answer
  5. Returns the questions and all of the answers
  """

  class QAState(TypedDict):
    topic: str
    questions: list[str]
    answers: Annotated[list[str], operator.add]  # each parallel branch appends its own answer

  def generate_questions(state: QAState) -> dict:
    response = LLM.invoke(
      f"Generate exactly 3 interesting questions about: {state['topic']}\n"
      "Format: numbered list, one question per line, no extra commentary."
    )
    questions = []
    for line in response.content.strip().splitlines():
      line = line.strip()
      if not line:
        continue
      # strip a leading "1. " / "2)" style marker so downstream prompts get a clean question
      questions.append(re.sub(r"^\d+[.)]\s*", "", line))
    return {"questions": questions}

  # 同一次觸發產生的目的地,無論幾個、無論用 Send 還是純名稱,都會被分進同一個 superstep
  def route_to_answers(state: QAState) -> list[Send]:
    # Number of questions is only known at runtime, so fan-out with Send
    # instead of a fixed edge per question.
    # 回傳 3 個 Send 後,LangGraph 會在同一個 superstep 平行派發、
    # 同時執行 3 次 Answer_question(不同 thread),而不是這行本身平行執行
    return [Send("Answer_question", {"question": q}) for q in state["questions"]]

  def answer_question(state: dict) -> dict:
    question = state["question"]
    response = LLM.invoke(f"Answer this question concisely:\n{question}")
    return {"answers": [f"Q: {question}\nA: {response.content}"]}

  # Create Graph, Add Nodes, Add Edges then Compile Graph
  graph = StateGraph(QAState)

  graph.add_node("Generate_questions", generate_questions)
  graph.add_node("Answer_question", answer_question)

  graph.add_edge(START, "Generate_questions")
  graph.add_conditional_edges("Generate_questions", route_to_answers, ["Answer_question"])
  graph.add_edge("Answer_question", END)

  app = graph.compile()

  # Save as PNG
  save_graph_png(app, "graph6_second_excercise.png")

  queries = ["The future of renewable energy", "The future of AI application, in Chinese"]
  for query in queries:
    result = app.invoke({"topic": query, "questions": [], "answers": []})

    print(f"\033[92mExercise Query: {query}\033[0m")
    print(f"\n\033[93mGenerated Questions:\n{result['questions']}\033[0m")
    for answer in result["answers"]:
      print(f"\n\033[38;5;208m{answer}\033[0m")
    print()


if __name__ == "__main__":
  print_section("Simple Graph")
  demo_simple_graph()

  # 成本/用量追蹤, 審計日誌(audit log), or 平行分支結果彙總
  print_section("Accumulating State")
  demo_accumulating_state()

  # go to .env LANGSMITH_TRACING=false to disable langsmith tracing for the next example, or set it to true to see the tracing in action
  # 任何聊天機器人 / 客服 assistant 的對話記憶
  print_section("Message State")
  demo_message_state()

  # 先分析、再擴充細節、最後收斂成摘要
  print_section("Multi-Node Graph")
  demo_multi_node_graph()

  # print_section("Exercise: First LangGraph")
  # exercise_first_langgraph()

  print_section("Exercise: Second LangGraph (Map-Reduce w/ Send)")
  exercise_second_langgraph()
