"""
Error Handling and Reliability Patterns
Building robust LangGraph applications
"""

import operator
import random
import time
from functools import wraps
from typing import Callable, Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")

MAX_RETRIES = 3  # total attempts (including the first call)
MAX_DELAY = 30.0  # max delay for retry backoff
BASE_DELAY = 1.0  # base delay for retry backoff
RANDOM_FAILURE_RATE = 0.5  # 50% chance of failure for demonstration
FAILURE_THRESHOLD = 3  # consecutive failures before circuit breaker opens
RECOVERY_TIMEOUT = 2.0  # seconds before an open circuit allows a half-open trial

FALLBACK_TIMEOUT = 10.0  # seconds for fallback chain model calls


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


# === Retry Decorator ===
# 參數	        預設值	   意義
# max_retries	  3	最多    嘗試幾次(含第一次)
# base_delay	  1.0 秒	  第一次失敗後基礎等待時間
# max_delay	    30.0 秒	  等待時間的上限,避免越等越久
# delay = min(base_delay * (2**attempt), max_delay)
def with_retry(
  max_retries: int = MAX_RETRIES,
  base_delay: float = BASE_DELAY,
  max_delay: float = MAX_DELAY,
  exceptions: tuple = (Exception,),
):
  """Retry decorator with exponential backoff."""

  def decorator(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
      last_exception = None

      for attempt in range(max_retries):
        try:
          return func(*args, **kwargs)
        except exceptions as e:
          last_exception = e
          if attempt < max_retries - 1:
            delay = min(base_delay * (2**attempt), max_delay)
            # Add jitter
            delay = delay * (0.5 + random.random())
            print(f"\033[33mAttempt {attempt + 1} failed:\033[0m {e}. Retrying in {delay:.1f}s")
            time.sleep(delay)
          else:
            print(f"\033[31mAttempt {attempt + 1} failed:\033[0m {e}. No more retries")

      raise last_exception

    return wrapper

  return decorator


@with_retry(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
def unreliable_api_call(query: str) -> str:
  """Simulates an unreliable API."""
  if random.random() < RANDOM_FAILURE_RATE:
    raise ConnectionError("Simulated API failure")
  return f"Success: {query}"


# Retry Pattern Demonstration
def demo_retry_pattern():
  """Demonstrate retry with exponential backoff."""

  for i in range(10):
    print(f"\033[92mQuery {i + 1}:\033[0m")
    try:
      result = unreliable_api_call(f"Query {i + 1}")
      print(f"✅ {result}")
    except Exception as e:
      print(f"\033[91m❌ Failed after retries: {e}\033[0m")
    print()


# === Circuit Breaker ===


class CircuitOpenError(Exception):
  """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
  """Circuit breaker pattern for failing services."""

  def __init__(
    self, failure_threshold: int = FAILURE_THRESHOLD, recovery_timeout: float = RECOVERY_TIMEOUT
  ):
    self.failure_threshold = failure_threshold
    self.recovery_timeout = recovery_timeout
    self.failures = 0
    self.last_failure_time = 0
    self.state = "closed"  # closed, open, half-open

  def call(self, func: Callable, *args, **kwargs):
    """Execute function with circuit breaker protection."""

    # Check if circuit should move from open to half-open
    if self.state == "open":
      # 過了恢復時間 → HALF-OPEN
      if time.time() - self.last_failure_time > self.recovery_timeout:
        print("   \033[33mCircuit breaker HALF-OPEN: trying one request\033[0m")
        self.state = "half-open"
      else:
        raise CircuitOpenError("Circuit breaker is OPEN")

    try:
      result = func(*args, **kwargs)

      # Success - close the circuit and reset the consecutive failure count
      if self.state == "half-open":
        print("   \033[32mCircuit breaker CLOSED after successful trial\033[0m")
      self.state = "closed"
      self.failures = 0

      return result

    except Exception as e:
      self.failures += 1
      self.last_failure_time = time.time()

      # A failed half-open trial reopens immediately; otherwise wait for the threshold
      # Service failures 達到門檻 → OPEN
      if self.state == "half-open" or self.failures >= self.failure_threshold:
        print(f"   \033[31mCircuit breaker OPENED after {self.failures} Service Failures\033[0m")
        self.state = "open"

      raise e


def demo_circuit_breaker():
  """Demonstrate circuit breaker pattern."""

  breaker = CircuitBreaker(failure_threshold=FAILURE_THRESHOLD, recovery_timeout=RECOVERY_TIMEOUT)

  def flaky_service():
    if random.random() < RANDOM_FAILURE_RATE:
      raise Exception("Service error")
    return "OK"

  for i in range(30):
    try:
      result = breaker.call(flaky_service)
      print(f"Attempt {i + 1}: ✅ {result} (state: {breaker.state})")
    except CircuitOpenError as e:
      print(f"Attempt {i + 1}: 🚫 {e} (state: {breaker.state})")
    except Exception as e:
      print(f"Attempt {i + 1}: ❌ {e} (state: {breaker.state})")

    time.sleep(0.5)


# === Model Fallback Chain ===
class FallbackChain:
  """Try multiple models in order until one succeeds."""

  def __init__(self):
    self.models = [
      ("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=FALLBACK_TIMEOUT)),
      ("gpt-4o", ChatOpenAI(model="gpt-4o", temperature=0, timeout=FALLBACK_TIMEOUT)),
      (
        "claude-sonnet",
        ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0, timeout=FALLBACK_TIMEOUT),
      ),
    ]
    self.cache = {}

  @traceable(name="fallback_invoke")
  def invoke(self, query: str, use_cache: bool = True) -> tuple[str, str]:
    """
    Invoke with fallbacks.
    Returns: (response, model_used)
    """

    # Check cache first
    if use_cache and query in self.cache:
      return self.cache[query], "cache"

    errors = []

    for model_name, model in self.models:
      try:
        response = model.invoke(query)
        result = response.content

        # Cache successful response
        self.cache[query] = result

        return result, model_name

      except Exception as e:
        errors.append(f"{model_name}: {str(e)}")
        continue

    # All models failed
    raise Exception(f"All models failed: {errors}")


# Fallback Chain Demonstration
def demo_fallback_chain():
  """Demonstrate fallback chain."""

  chain = FallbackChain()
  queries = [
    "What is 2 + 2?",
    "What is Python?",
    "What is the capital of France?",
  ]

  for i in range(2):  # Run multiple times to demonstrate caching # 2nd run should hit the cache
    for query in queries:
      try:
        result, model = chain.invoke(query)
        print(f"\033[92mQuery: {query}\033[0m")
        print(f"- Model: {model}")
        paragraphs = result.split("\n\n")
        paragraphs_length = len(paragraphs)
        first_paragraph = paragraphs[0]
        print(f"- Response: {first_paragraph}")
        if paragraphs_length > 1:
          print("  ...")
      except Exception as e:
        print(f"\033[91mQuery: {query}\033[0m")
        print(f"- ❌ Error: {e}")

      print()


# === LangGraph Error Handling ===
class RobustState(TypedDict):
  messages: Annotated[list, operator.add]
  error: Optional[str]
  retry_count: int
  max_retries: int
  success: bool
  simulated_failures: int  # how many times process should fail (demo only)


def create_robust_agent():
  """Create agent with built-in error handling."""

  def process_with_retry(state: RobustState) -> dict:
    """Process with retry logic built-in."""

    try:
      # Deterministic failure simulation: fail until retry_count reaches simulated_failures
      if state["retry_count"] < state["simulated_failures"]:
        raise Exception("Simulated processing error")

      response = LLM.invoke(state["messages"])

      return {"messages": [response], "success": True, "error": None}

    except Exception as e:
      return {
        "error": str(e),
        "retry_count": state["retry_count"] + 1,
        "success": False,
      }

  def should_continue(state: RobustState) -> Literal["retry", "error", "success"]:
    if state["success"]:
      return "success"
    elif state["retry_count"] < state["max_retries"]:
      return "retry"
    else:
      return "error"

  def handle_error(state: RobustState) -> dict:
    return {
      "messages": [
        AIMessage(
          content=f"I apologize, but I encountered an error: '{state['error']}'. "
          "Please try again later."
        )
      ]
    }

  def finalize(state: RobustState) -> dict:
    return state

  # Build graph
  graph = StateGraph(RobustState)

  graph.add_node("process", process_with_retry)
  graph.add_node("handle_error", handle_error)
  graph.add_node("finalize", finalize)

  graph.add_edge(START, "process")
  graph.add_conditional_edges(
    "process",
    should_continue,
    {"retry": "process", "error": "handle_error", "success": "finalize"},
  )
  graph.add_edge("handle_error", END)
  graph.add_edge("finalize", END)

  return graph.compile()


# Robust Agent Demonstration
def demo_robust_agent():
  """Demonstrate robust agent with error handling."""

  agent = create_robust_agent()
  save_graph_png(agent, "graphN_robust_agent_graph.png")

  # print("\nRobust Agent Demo:\n")

  # (label, simulated_failures) - max_retries is 3, so 3 failures exhaust the retries
  scenarios = [
    ("No failure -> finalize", 0),
    ("2 failures, then success -> retry x2 -> finalize", 2),
    ("Always fails -> retry x3 -> handle_error", 3),
  ]

  for i, (label, failures) in enumerate(scenarios):
    result = agent.invoke(
      {
        "messages": [HumanMessage(content="Hello!")],
        "error": None,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "success": False,
        "simulated_failures": failures,
      }
    )

    status = "✅ Success" if result["success"] else "❌ Failed"
    print(f"\n\033[92mScenario {i + 1} Label: {label}, Failures: {failures}:\033[0m")
    print(f"- Status: {status}")
    print(f"- Retries used: {result['retry_count']}")
    print(f"- Response: {result['messages'][-1].content}")


if __name__ == "__main__":
  # Example usage of the unreliable API call with retry logic
  # try:
  #   result = unreliable_api_call("Hello, World!")
  #   print(result)
  # except Exception as e:
  #   print(f"API call failed after retries: {e}")

  # # Run the retry pattern demonstration
  print_section("Retry Pattern Demo")
  demo_retry_pattern()

  # # Run the circuit breaker demonstration
  print_section("Circuit Breaker Demo")
  demo_circuit_breaker()

  # # Run the fallback chain demonstration
  print_section("Fallback Chain Demo")
  demo_fallback_chain()

  # # Run the robust agent demonstration
  print_section("Robust Agent Demo")
  demo_robust_agent()
