"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")

SYNTHESIZER_PROMPT = """Synthesize these three perspectives into a comprehensive response:

        RESEARCH: {research_result}
        CREATIVE: {creative_result}
        TECHNICAL: {technical_result}

        Create a unified, well-structured response."""

RESEARCH_PROMPT = """You are a research specialist. Gather facts and information relevant to the task. Be thorough but concise."""
CREATIVE_PROMPT = """You are a creative thinker. Provide novel perspectives and ideas."""
TECHNICAL_PROMPT = (
  """You are a technical analyst. Provide practical, implementation-focused insights."""
)

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert synthesizer. Combine multiple perspectives into coherent insights.  In Chinese"""
MAP_SYSTEM_PROMPT = """Summarize this document in 2-3 sentences. """
REDUCE_SYSTEM_PROMPT = """Combine these summaries into one coherent overview. Each summary should be represented in 1-2 sentences with a line break. The final summary should be concise and informative. In Chinese"""

PRINT_DEBUG = False  # Set to True to print debug information


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


class ParallelState(TypedDict):
  query: str
  research_result: str
  creative_result: str
  technical_result: str
  final_synthesis: str


def create_parallel_research():
  """Three research agents working in parallel."""

  def research_agent(state: ParallelState) -> dict:
    """Academic/factual research."""
    start = time.perf_counter()
    response = LLM.invoke(
      [
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=f"Research this topic: {state['query']}"),
      ]
    )
    print(f"\033[38;5;180m[research] took {time.perf_counter() - start:.2f}s\033[0m")
    return {"research_result": response.content}

  def creative_agent(state: ParallelState) -> dict:
    """Creative perspectives."""
    start = time.perf_counter()
    response = LLM.invoke(
      [
        SystemMessage(content=CREATIVE_PROMPT),
        HumanMessage(content=f"Give creative insights on: {state['query']}"),
      ]
    )
    print(f"\033[38;5;180m[creative] took {time.perf_counter() - start:.2f}s\033[0m")
    return {"creative_result": response.content}

  def technical_agent(state: ParallelState) -> dict:
    """Technical analysis."""
    start = time.perf_counter()
    response = LLM.invoke(
      [
        SystemMessage(content=TECHNICAL_PROMPT),
        HumanMessage(content=f"Analyze technically: {state['query']}"),
      ]
    )
    print(f"\033[38;5;180m[technical] took {time.perf_counter() - start:.2f}s\033[0m")
    return {"technical_result": response.content}

  def synthesize(state: ParallelState) -> dict:
    """Combine all perspectives."""
    synthesis_prompt = SYNTHESIZER_PROMPT.format(
      research_result=state["research_result"],
      creative_result=state["creative_result"],
      technical_result=state["technical_result"],
    )

    response = LLM.invoke(
      [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
      ]
    )
    return {"final_synthesis": response.content}

  graph = StateGraph(ParallelState)

  graph.add_node("research", research_agent)
  graph.add_node("creative", creative_agent)
  graph.add_node("technical", technical_agent)
  graph.add_node("synthesize", synthesize)

  # Fan-out: START goes to all three agents
  graph.add_edge(START, "research")
  graph.add_edge(START, "creative")
  graph.add_edge(START, "technical")

  graph.add_edge("research", "synthesize")
  graph.add_edge("creative", "synthesize")
  graph.add_edge("technical", "synthesize")

  graph.add_edge("synthesize", END)

  return graph.compile()


def demo_parallel_execution():
  """Demo parallel agent execution."""

  agent = create_parallel_research()
  save_graph_png(agent, "graphJ1_parallel_execution.png")

  queries = [
    "The future of remote work",
    "The future of AI software development.",
    # "The benefits of renewable energy",
  ]

  for query in queries:
    print(f"\n\033[32mQuery: {query}\033[0m")
    start = time.perf_counter()
    result = agent.invoke(
      {
        "query": query,
        "research_result": "",
        "creative_result": "",
        "technical_result": "",
        "final_synthesis": "",
      }
    )
    print(
      f"\n\033[33mTotal invoke time (all steps including synthesize): {time.perf_counter() - start:.2f}s\033[0m"
    )

    print("\n\033[32mIndividual Perspectives:\033[0m")
    print(f"\n\033[38;5;180m[Research]\033[0m\n{result['research_result']}")
    print(f"\n\033[38;5;180m[Creative]\033[0m\n{result['creative_result']}")
    print(f"\n\033[38;5;180m[Technical]\033[0m\n{result['technical_result']}")

    # print(f"\n{'=' * 50}")
    print(f"\n\033[33m[SYNTHESIZED]\033[0m\n{result['final_synthesis']}")


# Map-Reduce Pattern
class MapReduceState(TypedDict):
  documents: list[str]
  summaries: list[str]
  final_summary: str


def create_map_reduce_summarizer():
  """Summarize multiple documents in parallel."""

  def map_summarize(state: MapReduceState) -> dict:
    """Summarize each document (loop making one blocking LLM.invoke() after another)."""
    summaries = []
    for doc in state["documents"]:
      if PRINT_DEBUG:
        print(f"\n\033[38;5;180mDocument to be summarized:\033[0m\n{doc}\n")

      response = LLM.invoke(
        [
          SystemMessage(content=MAP_SYSTEM_PROMPT),
          HumanMessage(content=doc),
        ]
      )
      summaries.append(response.content)
    return {"summaries": summaries}

  def reduce_summaries(state: MapReduceState) -> dict:
    """Combine all summaries."""
    all_summaries = "\n\n".join([f"Summary {i + 1}: {s}" for i, s in enumerate(state["summaries"])])

    if PRINT_DEBUG:
      print(f"\n\033[38;5;180mAll summaries to be combined:\033[0m\n{all_summaries}\n")

    response = LLM.invoke(
      [
        SystemMessage(content=REDUCE_SYSTEM_PROMPT),
        HumanMessage(content=all_summaries),
      ]
    )
    return {"final_summary": response.content}

  graph = StateGraph(MapReduceState)
  graph.add_node("map", map_summarize)
  graph.add_node("reduce", reduce_summaries)

  graph.add_edge(START, "map")
  graph.add_edge("map", "reduce")
  graph.add_edge("reduce", END)

  return graph.compile()


def demo_map_reduce():
  """Demo map-reduce pattern."""

  agent = create_map_reduce_summarizer()
  save_graph_png(agent, "graphJ2_map_reduce.png")

  documents = [
    "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a vast ecosystem of libraries.",
    "Machine learning is a subset of AI that enables systems to learn from data. Common approaches include supervised, unsupervised, and reinforcement learning.",
    "Cloud computing provides on-demand access to computing resources. Major providers include AWS, Azure, and Google Cloud Platform.",
    "Renewable energy sources, such as solar and wind, are essential for reducing carbon emissions. They offer sustainable alternatives to fossil fuels.",
    "Blockchain technology enables secure and transparent transactions. It has applications in finance, supply chain, and digital identity management.",
  ]

  # print("\nMap-Reduce Summarization Demo:\n")

  result = agent.invoke({"documents": documents, "summaries": [], "final_summary": ""})

  print("\033[33mIndividual Map Summaries:\033[0m")
  for i, summary in enumerate(result["summaries"]):
    print(f"  \033[33m{i + 1}.\033[0m {summary}")

  print(f"\n\033[33mCombined Reduce Summary:\033[0m\n{result['final_summary']}")


if __name__ == "__main__":
  print_section("Demo Parallel Execution")
  demo_parallel_execution()

  print_section("Demo Map-Reduce Summarization")
  demo_map_reduce()
