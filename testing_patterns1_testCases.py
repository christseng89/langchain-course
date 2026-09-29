"""
Testing Patterns - T1 to T5
Unit tests (mocks), integration tests, LLM evaluation, regression testing
"""

from typing import Callable
from unittest.mock import Mock

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"[93mUsing LLM: {LLM.model_name}[0m")

PASS_THRESHOLD = 8


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


# === Unit Testing with Mocks ===
class QAChain:
  """Simple Q&A chain for testing."""

  def __init__(self, llm=None):
    self.llm = llm or LLM
    self.prompt = ChatPromptTemplate.from_template("Answer this question: {question}")

  def ask(self, question: str) -> str:
    prompt_value = self.prompt.invoke({"question": question})
    response = self.llm.invoke(prompt_value)
    return response.content


# The following unit tests use mocks to simulate LLM behavior without making real API calls.
# T1 Test the QAChain with a mocked LLM to ensure it returns the expected answer.
def test_qa_chain_with_mock():
  """Test QA chain with mocked LLM."""

  question = "What is the capital of France?"
  answer = "Paris"

  # Create mock LLM
  mock_llm = Mock()
  mock_llm.invoke.return_value = AIMessage(content=answer)

  # Test with mock
  chain = QAChain(llm=mock_llm)
  result = chain.ask(question)

  assert result == answer
  compare = result == answer
  mock_llm.invoke.assert_called_once()
  once = mock_llm.invoke.call_count == 1

  print("\033[92mQAChain with mock LLM:\033[0m")
  print(f"Question: {question}")
  print(f"Answer: {result}")
  print(f"Assert Mock: {compare}, Called Once: {once}")


# T2 Test the QAChain handles empty responses gracefully.
def test_qa_chain_handles_empty_response():
  """Test chain handles empty responses."""

  question = "Empty question"
  answer = ""

  mock_llm = Mock()
  mock_llm.invoke.return_value = AIMessage(content=answer)

  chain = QAChain(llm=mock_llm)
  result = chain.ask(question)

  assert result == answer
  compare = result == answer
  mock_llm.invoke.assert_called_once()
  once = mock_llm.invoke.call_count == 1

  print("\033[92mQAChain with mock LLM - Empty Response:\033[0m")
  print(f"Question: {question}")
  print(f"Answer: {result}")
  print(f"Assert Mock: {compare}, Called Once: {once}")


# === Integration Testing with Real LLM ===
class IntegrationTestSuite:
  """Integration tests with real LLM calls."""

  def __init__(self):
    self.llm = LLM

  @traceable(name="integration_test")
  def test_basic_qa(self) -> dict:
    """Test basic question answering."""

    test_cases = [
      {
        "question": "What is 2 + 2?",
        "expected_contains": ["4", "four"],
      },
      {
        "question": "What color is the sky on a clear day?",
        "expected_contains": ["blue"],
      },
    ]

    results = []
    for case in test_cases:
      response = self.llm.invoke(case["question"])
      content = response.content.lower()

      passed = any(exp.lower() in content for exp in case["expected_contains"])

      # "The answer is 4" or "2 + 2 equals four" or "That would be 4."

      results.append(
        {
          "question": case["question"],
          "response": response.content,
          "passed": passed,
        }
      )

    return {
      "total": len(results),
      "passed": sum(1 for r in results if r["passed"]),
      "results": results,
    }


# T3 Demonstrate integration tests with real LLM calls.
def demo_integration_tests():
  """Run integration tests."""

  suite = IntegrationTestSuite()
  results = suite.test_basic_qa()

  print(f"\033[92mPassed: {results['passed']}/{results['total']}\033[0m\n")

  for result in results["results"]:
    status = "✅" if result["passed"] else "❌"
    print(f"\033[93mQuestion: {result['question']}\033[0m")
    print(f"Status: {status}")
    print(f"Response:\n{result['response']}\n")


# === Evaluation Framework ===
class LLMEvaluator:
  """Use LLM to evaluate LLM outputs."""

  def __init__(self):
    self.llm = LLM

  # Traceable evaluation method for LangSmith logging -> evaluator.evaluate()
  @traceable(name="evaluate_response")
  def evaluate(self, question: str, response: str, reference: str = None) -> dict:
    """Evaluate a response on multiple dimensions."""

    eval_template = """
Evaluate this response on a scale of 1-10 for each criterion.

Question: {question}
Response: {response}
{reference_section}

Rate each criterion (1-10):
1. Correctness: Is the information accurate?
2. Relevance: Does it answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Does it fully address the question?

Respond with ONLY a JSON object:
{{"correctness": X, "relevance": X, "clarity": X, "completeness": X, "overall": X}}
"""
    eval_prompt = ChatPromptTemplate.from_template(eval_template)

    reference_section = ""
    if reference:
      reference_section = f"Reference answer: {reference}"

    import json

    response_obj = self.llm.invoke(
      eval_prompt.format(
        question=question,
        response=response,
        reference_section=reference_section,
      )
    )

    try:
      scores = json.loads(response_obj.content)
      return scores
    except json.JSONDecodeError:
      return {"error": "Failed to parse evaluation"}


# T4 Demonstrate LLM evaluation framework.
def demo_evaluation():
  """Demonstrate LLM evaluation."""

  evaluator = LLMEvaluator()

  # Test case 假装是AI產生的結果
  question = "Explain what machine learning is in simple terms."
  response = "Machine learning is when computers learn from data instead of being explicitly programmed. It's like teaching a child by showing examples rather than giving them rules."
  reference = "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions."

  print(f"\033[92mQuestion:\033[0m\n{question}\n")
  print(f"Response:\n{response}\n")
  print(f"Reference:\n{reference}")

  scores = evaluator.evaluate(question, response, reference)

  print("\033[93m\nScores:\033[0m")
  for metric, score in scores.items():
    print(f"- {metric}: {score}/10")


# === Regression Testing ===
class RegressionTestRunner:
  """Run regression tests against a test dataset."""

  def __init__(self, chain: Callable):
    self.chain = chain
    self.evaluator = LLMEvaluator()

  @traceable(name="regression_test")
  def run(self, test_cases: list[dict]) -> dict:
    """
    Run regression tests.

    test_cases: [{"input": ..., "expected": ...}, ...]
    """
    results = []
    total_score = 0

    for case in test_cases:
      # Get response from chain
      response = self.chain(case["input"])  # 真实由 AI 产生的 response

      # Evaluate
      scores = (
        self.evaluator.evaluate(  # 评判者即 T4 的 LLMEvaluator,再调用一次 LLM 对 response 打分
          question=case["input"],
          response=response,
          reference=case.get("expected"),
        )
      )

      overall = scores.get("overall", 0)
      total_score += overall

      results.append(
        {
          "input": case["input"],
          "response": response,
          "expected": case.get("expected"),
          "scores": scores,
          "passed": overall >= PASS_THRESHOLD,  # Threshold
        }
      )

    return {
      "total": len(results),
      "passed": sum(1 for r in results if r["passed"]),
      "average_score": total_score / len(results) if results else 0,
      "results": results,
    }


# T5 Demonstrate regression testing. T5 才是完整的"AI 评判 AI"
def demo_regression_testing():
  """Demonstrate regression testing."""

  # Simple chain to test
  # llm = LLM

  def qa_chain(question: str) -> str:
    return LLM.invoke(question).content

  # Test cases
  test_cases = [
    {
      "input": "What is Python?",
      "expected": "Python is a programming language known for its simplicity.",
    },
    {"input": "What is 10 * 5?", "expected": "50"},
    {
      "input": "What is the capital of Italy?",
      "expected": "Rome",
    },
    {
      "input": "Explain the benefits of exercise.",
      "expected": "Exercise improves physical and mental health.",
    },
  ]

  runner = RegressionTestRunner(qa_chain)
  results = runner.run(test_cases)

  print("\033[92mRegression Test Results:\033[0m")
  print(f"- Passed: {results['passed']}/{results['total']}")
  print(f"- Average Score: {results['average_score']:.1f}/10")

  for result in results["results"]:
    status = "✅" if result["passed"] else "❌"
    print(f"\033[93m\nQuestion: {result['input']}\033[0m")
    print(f"Expected: {result.get('expected', 'N/A')}\n")
    print(f"Response:\n{result['response']}\n")
    print(f"Status: {status}, Overall Score: {result['scores'].get('overall', 'N/A')}/10")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
  print_section("T1: Unit Tests with Mocks: QA Chain Returns Expected Answer")
  test_qa_chain_with_mock()

  print_section("T2: Unit Test with Mocks: QA Chain Handles Empty Response")
  test_qa_chain_handles_empty_response()

  print_section("T3: Integration Testing with Real LLM")
  demo_integration_tests()

  print_section("T4: LLM Evaluation Framework Demo - real judge, hardcoded response")
  demo_evaluation()

  print_section("T5: Regression Testing Demo - real judge, real LLM response")
  demo_regression_testing()

  print("\033[95mAll Tests Complete!\033[0m\n")
