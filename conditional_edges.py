from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

LLM = init_chat_model("gpt-4o-mini", temperature=0.0)
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


class RouterState(TypedDict):
  query: str
  query_type: str
  response: str


# Basic Routing
def demo_basic_routing():
  def classify_query(state: RouterState) -> dict:
    response = LLM.invoke(
      f"Classify this query as 'question', 'command', or 'statement'. "
      f"Reply with just the word.\n\n{state['query']}"
    )
    return {"query_type": response.content.lower().strip()}

  def handle_question(state: RouterState) -> dict:
    response = LLM.invoke(f"Answer this question: {state['query']}")
    return {"response": f"[Answer] {response.content}"}

  def handle_command(state: RouterState) -> dict:
    return {"response": f"[Executing] I'll help you with: {state['query']}"}

  def handle_statement(state: RouterState) -> dict:
    return {"response": f"[Acknowledged] Thanks for sharing: {state['query']}"}

  def route_by_type(
    state: RouterState,
  ) -> Literal["question", "command", "statement"]:
    qt = state["query_type"]
    if "question" in qt:
      return "question"
    elif "command" in qt:
      return "command"
    else:
      return "statement"

  graph = StateGraph(RouterState)

  graph.add_edge(START, "classify")
  graph.add_node("classify", classify_query)
  graph.add_node("handle_question", handle_question)
  graph.add_node("handle_command", handle_command)
  graph.add_node("handle_statement", handle_statement)

  graph.add_conditional_edges(
    "classify",  # source node
    route_by_type,  # function that determines which edge to take based on the state
    {
      "question": "handle_question",
      "command": "handle_command",
      "statement": "handle_statement",
    },
  )

  for node in ["handle_question", "handle_command", "handle_statement"]:
    graph.add_edge(node, END)

  app = graph.compile()
  save_graph_png(app, "graph9_basic.png")
  # # visualize the graph
  # print("\n--- Mermaid Graph ---")
  # print(app.get_graph().draw_mermaid())

  # # save as PNG
  # png_bytes = app.get_graph().draw_mermaid_png()
  # with open("graph_new.png", "wb") as f:
  #     f.write(png_bytes)
  # print("\nGraph saved to graph_new.png")

  # Example usage
  queries = [
    "What is the capital of France?",
    "Send an email to John",
    "I love programming",
  ]

  for query in queries:
    result = app.invoke({"query": query})
    print(f"\033[92m\nQuery: {query}\033[0m")
    print(f"Type: {result['query_type']}")
    print(f"Response: {result['response']}")
    # print("-" * 40)


class QualityState(TypedDict):
  content: str
  quality_score: int
  feedback: str
  final_content: str
  iteration: int


# Conditional Loop DEMO
def demo_conditional_loop():

  def evaluate_quality(state: QualityState) -> dict:
    response = LLM.invoke(
      f"Rate this content quality from 1-10. Reply with just the number.\n\n"
      f"Content: {state['content']}"
    )
    try:
      score = int(response.content.strip())
    except Exception:
      score = 5
    return {"quality_score": score}

  def improve_content(state: QualityState) -> dict:
    response = LLM.invoke(
      f"Improve this content to be more engaging and clear:\n\n{state['content']}"
    )
    return {"content": response.content, "iteration": state["iteration"] + 1}

  def finalize_content(state: QualityState) -> dict:
    return {
      "final_content": state["content"],
      "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}",
    }

  def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
    if state["quality_score"] >= 8 or state["iteration"] >= 3:
      return "finalize"  # score threshold met, or max iterations reached
    else:
      return "improve"

  graph = StateGraph(QualityState)

  graph.add_edge(START, "evaluate")
  graph.add_node("evaluate", evaluate_quality)
  graph.add_node("improve", improve_content)
  graph.add_node("finalize", finalize_content)

  graph.add_conditional_edges(
    "evaluate", should_continue, {"improve": "improve", "finalize": "finalize"}
  )

  graph.add_edge("improve", "evaluate")  # Loop back!
  graph.add_edge("finalize", END)

  app = graph.compile()
  save_graph_png(app, "graphA_conditional.png")

  # # visualize the graph
  # print("\n--- Mermaid Graph ---")
  # print(app.get_graph().draw_mermaid())

  # # save as PNG
  # png_bytes = app.get_graph().draw_mermaid_png()
  # with open("graph_newest.png", "wb") as f:
  #   f.write(png_bytes)
  # print("\nGraph saved to graph_newest.png")

  # Example usage
  # print("\nConditional Loop Demo:\n")

  contents = [
    "AI is cool",
    "it good",
    "Our AI-powered platform streamlines your workflow, saving you hours every week while reducing costly errors.",
  ]

  for content in contents:
    # content = "AI is cool"
    result = app.invoke(
      {
        "content": content,
        "quality_score": 0,
        "feedback": "",
        "final_content": "",
        "iteration": 0,
      }
    )

    print(f"\033[92m\nContent: {content}\033[0m")
    print(f"Final: {result['final_content']}")
    print(f"\033[38;5;208mFeedback: {result['feedback']}\033[0m")


# Multi Path Routing DEMO
def demo_multi_path_routing():
  class TaskState(TypedDict):
    task: str
    urgency: str
    complexity: str
    handler: str
    result: str

  def analyze_task(state: TaskState) -> dict:
    # Analyze urgency
    urgency_response = LLM.invoke(
      f"Is this task urgent? Reply 'urgent' or 'normal'.\nTask: {state['task']}"
    )

    # Analyze complexity
    complexity_response = LLM.invoke(
      f"Is this task complex? Reply 'complex' or 'simple'.\nTask: {state['task']}"
    )

    return {
      "urgency": urgency_response.content.lower().strip(),
      "complexity": complexity_response.content.lower().strip(),
    }

  def urgent_complex_handler(state: TaskState) -> dict:
    return {
      "handler": "Senior Team",
      "result": "Escalated to senior team for immediate action",
    }

  def urgent_simple_handler(state: TaskState) -> dict:
    return {
      "handler": "Quick Response",
      "result": "Handled immediately by available agent",
    }

  def normal_complex_handler(state: TaskState) -> dict:
    return {
      "handler": "Specialist",
      "result": "Assigned to specialist for thorough handling",
    }

  def normal_simple_handler(state: TaskState) -> dict:
    return {
      "handler": "Standard",
      "result": "Added to standard queue",
    }

  def route_task(
    state: TaskState,
  ) -> Literal["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
    is_urgent = "urgent" in state["urgency"]
    is_complex = "complex" in state["complexity"]

    if is_urgent and is_complex:
      return "urgent_complex"
    elif is_urgent:
      return "urgent_simple"
    elif is_complex:
      return "normal_complex"
    else:
      return "normal_simple"

  graph = StateGraph(TaskState)

  graph.add_edge(START, "analyze")
  graph.add_node("analyze", analyze_task)
  graph.add_node("urgent_complex", urgent_complex_handler)
  graph.add_node("urgent_simple", urgent_simple_handler)
  graph.add_node("normal_complex", normal_complex_handler)
  graph.add_node("normal_simple", normal_simple_handler)

  graph.add_conditional_edges(
    "analyze",
    route_task,
    {
      "urgent_complex": "urgent_complex",
      "urgent_simple": "urgent_simple",
      "normal_complex": "normal_complex",
      "normal_simple": "normal_simple",
    },
  )

  for node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
    graph.add_edge(node, END)

  app = graph.compile()

  save_graph_png(app, "graphB_complex.png")
  # # visualize the graph
  # print("\n--- Mermaid Graph ---")
  # print(app.get_graph().draw_mermaid())

  # # save as PNG
  # png_bytes = app.get_graph().draw_mermaid_png()
  # with open("graph_complex.png", "wb") as f:
  #   f.write(png_bytes)
  # print("\nGraph saved to graph_complex.png")

  # print("\nMulti-Path Routing Demo:\n")

  tasks = [
    "Server is down! Need immediate fix!",
    "Update the documentation for the API",
    "Redesign the entire database schema",
    "Fix the typo on the homepage",
    "Bug fixing",
    "Enhancement",
  ]

  for task in tasks:
    print(f"\033[92m\nTask: {task}\033[0m")

    result = app.invoke({"task": task})
    print(f"Urgency: {result['urgency']} | Complexity: {result['complexity']}")
    print(f"Handler: {result['handler']}")
    print(f"\033[38;5;208mResult: {result['result']}\033[0m")
    # print("-" * 40)


if __name__ == "__main__":
  print_section("Basic Literal Routing")
  demo_basic_routing()

  print_section("Conditional Literal Loop")
  demo_conditional_loop()

  print_section("Multi-Path Routing")
  demo_multi_path_routing()
