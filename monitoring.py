"""
Monitoring and Logging for Production
Structured logging, metrics, and alerts
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

from semantic_cache import SemanticCache

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


# === Structured Logging ===
class JSONFormatter(logging.Formatter):
  """Format logs as JSON for log aggregation."""

  def format(self, record):
    log_obj = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "level": record.levelname,
      "message": record.getMessage(),
      "module": record.module,
      "function": record.funcName,
    }

    if hasattr(record, "extra_data"):
      log_obj.update(record.extra_data)

    return json.dumps(log_obj)


def setup_logging():
  """Setup structured JSON logging."""

  logger = logging.getLogger("langgraph_app")
  logger.setLevel(logging.INFO)

  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(JSONFormatter())
  logger.addHandler(handler)

  return logger


# === Metrics Collection ===
class MetricsCollector:
  """Collect and aggregate metrics."""

  def __init__(self):
    self.metrics = {
      "requests_total": 0,
      "errors_total": 0,
      "latency_sum": 0,
      "latency_count": 0,
      "tokens_input": 0,
      "tokens_output": 0,
      "cache_hits": 0,
      "cache_misses": 0,
    }

  def record_request(
    self,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    error: bool = False,
    cache_hit: bool = False,
  ):
    self.metrics["requests_total"] += 1
    self.metrics["latency_sum"] += latency_ms
    self.metrics["latency_count"] += 1
    self.metrics["tokens_input"] += input_tokens
    self.metrics["tokens_output"] += output_tokens

    if error:
      self.metrics["errors_total"] += 1

    if cache_hit:
      self.metrics["cache_hits"] += 1
    else:
      self.metrics["cache_misses"] += 1

  def get_summary(self) -> dict:
    avg_latency = (
      self.metrics["latency_sum"] / self.metrics["latency_count"]
      if self.metrics["latency_count"] > 0
      else 0
    )
    error_rate = (
      self.metrics["errors_total"] / self.metrics["requests_total"]
      if self.metrics["requests_total"] > 0
      else 0
    )
    cache_hit_rate = (
      self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
      if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
      else 0
    )

    return {
      "total_requests": self.metrics["requests_total"],
      "total_errors": self.metrics["errors_total"],
      "error_rate": f"{error_rate:.2%}",
      "avg_latency_ms": round(avg_latency, 2),
      "total_input_tokens": self.metrics["tokens_input"],
      "total_output_tokens": self.metrics["tokens_output"],
      "cache_hit_rate": f"{cache_hit_rate:.2%}",
    }


# === Instrumented LLM ===
class InstrumentedLLM:
  """LLM with full instrumentation."""

  def __init__(self):
    self.llm = LLM
    self.cache = SemanticCache()
    self.metrics = MetricsCollector()
    self.logger = setup_logging()
    self.last_request_stats = {}

  def log_last_request(self):
    """Log the stats of the most recent successful request."""
    self.logger.info("LLM request completed", extra={"extra_data": self.last_request_stats})

  @traceable(name="instrumented_invoke")
  def invoke(self, query: str) -> str:
    start_time = time.time()
    error = False

    try:
      # Check cache first; a hit costs no tokens
      cached = self.cache.get(query)
      if cached:
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.record_request(
          latency_ms=latency_ms,
          input_tokens=0,
          output_tokens=0,
          error=error,
          cache_hit=True,
        )
        self.last_request_stats = {
          "latency_ms": round(latency_ms, 2),
          "input_tokens": 0,
          "output_tokens": 0,
          "cache_hit": True,
        }
        return cached

      response = self.llm.invoke(query)
      result = response.content
      self.cache.set(query, result)

      # Estimate tokens
      input_tokens = len(query.split()) * 4 // 3
      output_tokens = len(result.split()) * 4 // 3

      self.metrics.record_request(
        latency_ms=(time.time() - start_time) * 1000,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
        cache_hit=False,
      )

      self.last_request_stats = {
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit": False,
      }

      return result

    except Exception as e:
      error = True
      self.metrics.record_request(
        latency_ms=(time.time() - start_time) * 1000,
        input_tokens=0,
        output_tokens=0,
        error=error,
        cache_hit=False,
      )

      self.logger.error(f"LLM request failed: {e}", extra={"extra_data": {"error": str(e)}})

      raise


# Demonstration of monitoring
def demo_monitoring():
  """Demonstrate monitoring."""

  llm = InstrumentedLLM()

  # print("Monitoring Demo:\n")

  queries = [
    "What is Python?",
    "Explain machine learning.",
    "What is 2 + 2?",
    "What is Python?",  # Duplicate query to test caching
    "What is 2 + 2?",  # Duplicate query to test caching
  ]

  for i, query in enumerate(queries):
    result = llm.invoke(query)
    print(f"\n\033[92mQuery {i + 1}: {query}\033[0m")
    response = result.split("\n\n")  # Show only first line of response
    if len(response) > 1:
      response = response[0] + " (truncated)..."
    else:
      response = response[0]
    print(f"Result: {response}")

    print("\n\033[94mLog:\033[0m ", end="")
    llm.log_last_request()

  print("\n\033[93mMetrics Summary:\033[0m")
  summary = llm.metrics.get_summary()
  for key, value in summary.items():
    print(f"- {key}: {value}")


if __name__ == "__main__":
  # logger = setup_logging()
  # logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})
  print_section("Monitoring and Logging Demo")
  demo_monitoring()
