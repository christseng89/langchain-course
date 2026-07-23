import operator
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv()

"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

LLM = init_chat_model("gpt-4o-mini", temperature=0.0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")

PRINT_FINDING_ALL = False
MAX_ITERATIONS1 = 5  # Self-Correcting Code
MAX_ITERATIONS2 = 3  # Iterative Research


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


class TestPlan(BaseModel):
  function_name: str = Field(description="Exact name the generated function must use")
  test_code: list[str] = Field(
    description="Standalone Python `assert` statements that call the function by "
    "name and check correct behavior. Only use deterministic checks (avoid "
    "assertions that depend on the current date/time or system locale)."
  )


class CodeGenState(TypedDict):
  task: str
  function_name: str
  test_cases: list[str]
  code: str
  errors: Annotated[list[str], operator.add]
  iteration: int
  max_iterations: int
  success: bool


# Self Correcting Code DEMO
def demo_self_correcting_code():
  """Self-correcting code generator."""

  def generate_test_cases(state: CodeGenState) -> dict:
    plan_llm = LLM.with_structured_output(TestPlan)
    plan = plan_llm.invoke(
      f"Task: {state['task']}\n\n"
      "Design a test plan for a Python function that solves this task: "
      "pick a function name, then write 3-4 standalone `assert` statements "
      "that call the function and check its behavior."
    )
    return {"function_name": plan.function_name, "test_cases": plan.test_code}

  def generate_code(state: CodeGenState) -> dict:
    if state["iteration"] == 0:
      # First attempt
      prompt = (
        f"Write Python code for: {state['task']}\n"
        f"The function must be named exactly '{state['function_name']}'.\n"
        "Return only the code."
      )
    else:
      # Correction attempt
      prompt = (
        f"Fix this Python code:\n{state['code']}\n\n"
        f"Errors:\n{state['errors'][-1]}\n\n"
        f"The function must still be named exactly '{state['function_name']}'.\n"
        "Return only the corrected code."
      )

    response = LLM.invoke(prompt)
    code = response.content.strip()

    # Clean up markdown code blocks if present
    if code.startswith("```"):
      code = code.split("```")[1]
      if code.startswith("python"):
        code = code[6:]

    return {"code": code, "iteration": state["iteration"] + 1}

  def validate_code(state: CodeGenState) -> dict:
    code = state["code"]

    # Step 1: Does it compile?
    try:
      compile(code, "<string>", "exec")
    except SyntaxError as e:
      return {"errors": [f"SyntaxError: {e}"], "success": False}

    # Step 2: Does it RUN and produce correct results?
    namespace = {}
    try:
      exec(code, namespace)
    except Exception as e:
      return {"errors": [f"Runtime error: {e}"], "success": False}

    if state["function_name"] not in namespace:
      return {
        "errors": [f"Function '{state['function_name']}' not found in code"],
        "success": False,
      }

    failures = []
    for test in state["test_cases"]:
      try:
        exec(test, namespace)
      except AssertionError:
        failures.append(f"Assertion failed: {test}")
      except Exception as e:
        failures.append(f"{test} raised {e}")

    if failures:
      return {"errors": ["\n".join(failures)], "success": False}

    return {"success": True}

  def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
    if state["success"]:
      return "end"
    elif state["iteration"] >= state["max_iterations"]:
      return "end"
    else:
      return "generate"

  def finalize(state: CodeGenState) -> dict:
    return state

  graph = StateGraph(CodeGenState)

  graph.add_edge(START, "generate_test_cases")
  graph.add_node("generate_test_cases", generate_test_cases)
  graph.add_node("generate", generate_code)
  graph.add_node("validate", validate_code)
  graph.add_node("finalize", finalize)

  graph.add_edge("generate_test_cases", "generate")
  graph.add_edge("generate", "validate")
  graph.add_conditional_edges(
    "validate", should_continue, {"generate": "generate", "end": "finalize"}
  )  # Loop back to "generate" if not successful and under max iterations, otherwise go to "finalize"
  graph.add_edge("finalize", END)

  app = graph.compile()
  save_graph_png(app, "graphC_self_correction.png")
  # # visualize the graph
  # print("\n--- Mermaid Graph ---")
  # # print(app.get_graph().draw_mermaid())

  # # save as PNG
  # png_bytes = app.get_graph().draw_mermaid_png()
  # with open("graph_code.png", "wb") as f:
  #   f.write(png_bytes)
  # print("\nGraph saved to graph_code.png")

  # print("Self-Correcting Code Generator:\n")

  tasks = [
    "a function that calculates factorial recursively",
    "a function that converts a yyyy-mm-dd formatted date string into dd/mm/yyyy format",
  ]

  for task in tasks:
    result = app.invoke(
      {
        "task": task,
        "function_name": "",
        "test_cases": [],
        "code": "",
        "errors": [],
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS1,
        "success": False,
      }
    )

    print(f"\n\033[92mTask: \033[0m{result['task']}")
    print(f"Function name: {result['function_name']}")
    print("\033[38;5;208m\nTest Cases:\033[0m\n")
    for i, test_case in enumerate(result["test_cases"]):
      print(f"{i + 1}. {test_case}")
    print(f"\nIterations: {result['iteration']}")
    print(f"\033[93mSuccess: {result['success']}\033[0m")
    print(f"\033[38;5;208m\nFinal Code:\033[0m\n{result['code']}")


class ResearchState(TypedDict):
  topic: str
  findings: Annotated[list[str], operator.add]
  questions: list[str]
  iteration: int
  max_iterations: int
  summary: str


# Interative Research DEMO
def demo_iterative_research():
  """Iterative research that goes deeper based on findings."""

  def research(state: ResearchState) -> dict:
    # print(f"\n{'─' * 50}")

    depth = f"{state['iteration'] + 1}/{state['max_iterations']}"

    if state["iteration"] == 0:
      print(f"\n\033[92mTopic: {state['topic']}\033[0m")
      print(f"\n\033[38;5;130m📚 [RESEARCH] Starting on depth: {depth}\033[0m")
      query = f"Give me 3 key facts about: {state['topic']}"
      # print(f"\033[93m fresh on: {state['topic']}\033[0m")
    else:
      question = state["questions"][-1] if state["questions"] else "elaborate"
      query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper: {question}"
      print(f"Follow up on depth: {depth}\nFollow up question: \n{question}")

    response = LLM.invoke(query)
    print(f"\n✅ Found {len(response.content.splitlines())} lines of findings")
    if PRINT_FINDING_ALL:
      print(f"🔍 Findings: {response.content}")
    else:
      print(f"🔍 Findings preview: {response.content[:120]}...")
    return {"findings": [response.content]}

  def generate_questions(state: ResearchState) -> dict:
    # print(f"\n{'─' * 50}")
    depth = f"{state['iteration'] + 1}/{state['max_iterations']}"
    print(f"\033[38;5;130m\n🤔 [QUESTIONING] Analyzing latest findings on depth {depth}...\033[0m")

    response = LLM.invoke(
      f"Based on this finding:\n{state['findings'][-1]}\n\n"
      "What's one deeper question to explore? Reply with just the question."
    )

    print(f"Next question: {response.content.strip()}")
    return {"questions": [response.content], "iteration": state["iteration"] + 1}

  def synthesize(state: ResearchState) -> dict:
    # print(f"\n{'─' * 50}")
    print(f"🧬 [SYNTHESIZE] Combining {len(state['findings'])} rounds of findings...")

    all_findings = "\n\n".join(state["findings"])
    response = LLM.invoke(f"Synthesize these findings into a coherent summary:\n\n{all_findings}")

    print(f"✅ Summary generated ({len(response.content.split())} words)")
    return {"summary": response.content}

  def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
    depth = f"{state['iteration']}/{state['max_iterations']}"

    if state["iteration"] >= state["max_iterations"]:
      print(f"\033[38;5;130m\n🏁 [ROUTER] Max depth reached ({depth}) → synthesizing\033[0m")
      return "synthesize"
    print(f"\033[38;5;130m\n🔄 [ROUTER] Depth {depth} → going deeper\033[0m")
    return "research"

  graph = StateGraph(ResearchState)

  graph.add_node("research", research)
  graph.add_node("generate_questions", generate_questions)
  graph.add_node("synthesize", synthesize)

  graph.add_edge(START, "research")
  graph.add_edge("research", "generate_questions")
  graph.add_conditional_edges(
    "generate_questions",
    should_continue,
    {"research": "research", "synthesize": "synthesize"},
  )
  graph.add_edge("synthesize", END)

  app = graph.compile()

  save_graph_png(app, "graphD_interative_research.png")
  # print("=" * 50)
  # print("🔬 ITERATIVE RESEARCH WORKFLOW")
  # print("=" * 50)

  topics = [
    "quantum computing applications",
    "Agentic AI applications",
    "Next generation Trade Finance Solution for Banks",
  ]

  for topic in topics:
    result = app.invoke(
      {
        "topic": topic,
        "findings": [],
        "questions": [],
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS2,
        "summary": "",
      }
    )

    # print(f"\n{'=' * 50}")
    print(f"\033[93m\n📊 Topic: {result['topic']} Research Complete\033[0m")

    print(f"Depth reached: {result['iteration']}")
    print(f"Findings collected: {len(result['findings'])}")
    print(f"Questions explored: {len(result['questions'])}")
    print(f"\033[92m\n📝 Final Summary:\033[0m\n{result['summary']}")


if __name__ == "__main__":
  print_section("🔧 Self-Correcting Code")
  demo_self_correcting_code()

  print_section("🔬 Iterative Research")
  demo_iterative_research()
