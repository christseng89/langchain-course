"""
Cost Optimization Patterns
Reducing LLM costs in production
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

from semantic_cache import SemanticCache

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")

LLM1 = ChatOpenAI(model="gpt-4o", temperature=0)
print(f"\033[93mUsing LLM: {LLM1.model_name}\033[0m")

MAX_TOKENS_PER_REQUEST = 4000
MAX_TOKENS = 100


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


# === Model Routing ===
class ModelRouter:
  """Route queries to appropriate model based on complexity."""

  def __init__(self):
    self.cheap_model = LLM
    self.expensive_model = LLM1
    self.classifier = LLM

  def classify_complexity(self, query: str) -> str:
    """Classify query complexity."""

    prompt = ChatPromptTemplate.from_template(
      """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: Analysis, reasoning, creative tasks, multi-step problems

Query: {query}

Respond with only: simple or complex
"""
    )

    response = self.classifier.invoke(prompt.format(query=query))
    return response.content.strip().lower()

  @traceable(name="routed_query")
  def invoke(self, query: str) -> tuple[str, str, float]:
    """
    Route and invoke query.
    Returns: (response, model_used, estimated_cost)
    """
    complexity = self.classify_complexity(query)

    if complexity == "simple":
      model = self.cheap_model
      model_name = "gpt-4o-mini"
      cost_per_1k = 0.00015  # Input cost
    else:
      model = self.expensive_model
      model_name = "gpt-4o"
      cost_per_1k = 0.0025  # Input cost

    response = model.invoke(query)

    # Estimate cost (rough)
    tokens = len(query.split()) * 1.3  # Rough token estimate
    estimated_cost = (tokens / 1000) * cost_per_1k

    return response.content, model_name, estimated_cost


def demo_model_routing():
  """Demonstrate model routing."""

  router = ModelRouter()

  queries = [
    "What is 2 + 2?",  # Simple
    "Analyze the economic implications of AI on the job market.",  # Complex
    "What color is the sky?",  # Simple
  ]

  # print("Model Routing Demo:\n")

  total_cost = 0
  for query in queries:
    result, model, cost = router.invoke(query)
    total_cost += cost
    print(f"\033[32mQuery: {query}\033[0m")
    print(f"\033[33m- Model: {model}\033[0m")
    response = result.split("\n\n")  # Show only first line of response
    if len(response) > 1:
      response = response[0] + " (truncated)..."
    else:
      response = response[0]
    print(f"\033[32m- Response: {response}\033[0m")
    print(f"\033[33m- Est. Cost: ${cost:.6f}\033[0m\n")

  print(f"\033[34mTotal Estimated Cost: ${total_cost:.6f}\033[0m")


# === Semantic Caching ===
# SemanticCache lives in semantic_cache.py
class CachedLLM:
  """LLM wrapper with caching."""

  def __init__(self):
    self.llm = LLM
    self.cache = SemanticCache()
    self.cache_hits = 0
    self.cache_misses = 0

  @traceable(name="cached_invoke")
  def invoke(self, query: str) -> tuple[str, bool]:
    """
    Invoke with caching.
    Returns: (response, from_cache)
    """
    # Check cache
    cached = self.cache.get(query)
    if cached:
      self.cache_hits += 1
      return cached, True

    # Call LLM
    self.cache_misses += 1
    response = self.llm.invoke(query)
    result = response.content

    # Cache result
    self.cache.set(query, result)

    return result, False

  def get_stats(self) -> dict:
    total = self.cache_hits + self.cache_misses
    hit_rate = self.cache_hits / total if total > 0 else 0
    return {
      "hits": self.cache_hits,
      "misses": self.cache_misses,
      "hit_rate": f"{hit_rate:.1%}",
    }


# Demonstration of caching
def demo_caching():
  """Demonstrate caching."""

  llm = CachedLLM()

  queries = [
    "What is Python?",
    "What is JavaScript?",
    "What is Python?",  # Cache hit
    "What is python?",  # Cache hit (normalized)
    "What is Rust?",
    "What is Java",
  ]

  # print("\nCaching Demo:\n")

  for query in queries:
    print(f"\033[32mQuery: {query}\033[0m")
    result, from_cache = llm.invoke(query)
    source = "CACHE" if from_cache else "LLM"
    print(f"\033[33m- Source: [{source}]\033[0m")
    response = result.split("\n\n")  # Show only first line of response
    if len(response) > 1:
      response = response[0] + " (truncated)..."
    else:
      response = response[0]

    print(f"- Response: {response}\n")

  print(f"\n\033[34mStats: {llm.get_stats()}\033[0m")


# === Token Budgeting ===
class TokenBudget:
  """Track and limit token usage."""

  def __init__(self, max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST):
    self.max_per_request = max_tokens_per_request
    self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

  def estimate_tokens(self, text: str) -> int:
    """Rough token estimation (actual would use tiktoken)."""
    return int(len(text.split()) * 1.3)

  def check_budget(self, text: str) -> tuple[bool, int]:
    """Check if request is within budget."""
    tokens = self.estimate_tokens(text)
    # print(f"\033[38;5;94mEstimated tokens for request: {tokens}\033[0m")
    return tokens <= self.max_per_request, tokens

  def record_usage(self, input_tokens: int, output_tokens: int):
    """Record token usage."""
    self.usage["total_input"] += input_tokens
    self.usage["total_output"] += output_tokens
    self.usage["requests"] += 1

  def get_stats(self) -> dict:
    return {
      **self.usage,
      "total_tokens": self.usage["total_input"] + self.usage["total_output"],
      "avg_per_request": (
        (self.usage["total_input"] + self.usage["total_output"]) / max(self.usage["requests"], 1)
      ),
    }


class BudgetedLLM:
  """LLM with token budgeting."""

  def __init__(self, max_tokens: int = MAX_TOKENS_PER_REQUEST):
    self.llm = LLM
    self.budget = TokenBudget(max_tokens_per_request=max_tokens)

  @traceable(name="budgeted_invoke")
  def invoke(self, query: str) -> str:
    # Check budget
    within_budget, tokens = self.budget.check_budget(query)

    # Raise error if over budget
    if not within_budget:
      raise ValueError(f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}")

    # Execute
    response = self.llm.invoke(query)
    result = response.content

    # Record usage
    output_tokens = self.budget.estimate_tokens(result)
    self.budget.record_usage(tokens, output_tokens)

    return result

  def get_stats(self) -> dict:
    return self.budget.get_stats()


def demo_token_budgeting():
  """Demonstrate token budgeting."""

  llm = BudgetedLLM(max_tokens=MAX_TOKENS)

  queries = [
    "What is AI?",  # Within budget
    "Explain " + "very " * 10 + "\n" + "very " * 90 + "complex topic",  # Over budget
    "What is the capital of France?",  # Within budget
    "Write a short poem about the sea.",  # Within budget
  ]

  # print("\nToken Budgeting Demo:\n")

  for query in queries:
    try:
      result = llm.invoke(query)
      print(f"\033[32m✅ Query: {query}\033[0m")
      response = result.split("\n\n")  # Show only first line of response
      if len(response) > 1:
        response = response[0] + " (truncated)..."
      else:
        response = response[0]
      print(f"\033[33mResponse:\033[0m\n{response}\n")
    except ValueError as e:
      question = query.split("\n")[0] + " (truncated)..."  # Show only first line of query
      print(f"\033[31m❌ Query: {question}\033[0m")
      print(f"Error: {e}\n")

  print(f"\033[34mStats: {llm.get_stats()}\033[0m")


if __name__ == "__main__":
  print_section("Demo Model Routing")
  demo_model_routing()

  print_section("DemoCaching")
  demo_caching()

  print_section("Demo Token Budgeting")
  demo_token_budgeting()

  # Production version would:
# 1. Embed the query into a vector
# 2. Search the cache by vector similarity
# 3. Return if similarity > threshold (e.g., 0.95)
