"""
LangSmith Evaluation - Production Approach (Step 1 to Step 3)
Persistent, versioned datasets, experiments and comparisons
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from langsmith.evaluation import evaluate  # Using the evaluate function from langsmith.evaluation

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"[93mUsing LLM: {LLM.model_name}[0m")

client = Client()


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


def print_results(results: list[dict], version: str) -> None:
  """Print evaluation results in a readable format."""
  print(f"\n\033[92mEvaluation Results ({version}):\033[0m")
  for result in results:
    question = result["example"].inputs.get("question", "N/A")
    answer = result["run"].outputs.get("answer", "N/A")

    print(f"\n\033[93mQuestion: {question}\033[0m")
    print(f"AI Answer:\n{answer}\n")

    print("\033[33mScores:\033[0m")
    for eval_result in result["evaluation_results"]["results"]:
      print(f"- {eval_result.key}: {eval_result.score}")


# ============================================================
# Step 1: Create an evaluation dataset
# ============================================================
"""
LangSmith Evaluation Datasets — Production Approach
Persistent, versioned test suites for LLM applications
"""


def create_eval_dataset():
  """Create a dataset with test cases in LangSmith."""

  dataset_name = "qa-eval-dataset"

  # Delete if exists (for demo purposes — don't do this in production)
  existing = list(client.list_datasets(dataset_name=dataset_name))
  if existing:
    client.delete_dataset(dataset_id=existing[0].id)

  dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Q&A evaluation dataset for testing our chain",
  )

  # Add test examples — inputs and expected outputs
  examples = [
    {
      "inputs": {"question": "What is Python?"},
      "outputs": {
        "answer": "Python is a high-level programming language known for its readability and versatility."
      },
    },
    {"inputs": {"question": "What is 15 * 4?"}, "outputs": {"answer": "60"}},
    {
      "inputs": {"question": "What does HTML stand for?"},
      "outputs": {"answer": "HyperText Markup Language"},
    },
    {
      "inputs": {"question": "Name one benefit of exercise."},
      "outputs": {
        "answer": "Exercise improves cardiovascular health and reduces the risk of chronic diseases."
      },
    },
    {
      "inputs": {"question": "What is the capital of Japan?"},
      "outputs": {"answer": "Tokyo"},
    },
  ]

  for example in examples:
    print(
      f"\033[93mExample: {example['inputs']['question']}\033[0m\n→ {example['outputs']['answer']}\n"
    )
    client.create_example(
      inputs=example["inputs"], outputs=example["outputs"], dataset_id=dataset.id
    )

  print(f"\033[92mCreated dataset '{dataset_name}' with {len(examples)} examples\033[0m")
  return dataset_name


prompt = ChatPromptTemplate.from_template("Answer this question concisely: {question}")
v1_chain = prompt | LLM


# QA target function for LangSmith evaluation. Must accept a dict (inputs) and return a dict (outputs).
@traceable(name="qa_target_v1")
def qa_target_v1(inputs: dict) -> dict:
  """
  Target function for LangSmith evaluation.
  Must accept a dict (inputs) and return a dict (outputs).
  """
  response = v1_chain.invoke({"question": inputs["question"]})
  return {"answer": response.content}


# ============================================================
# Define evaluators
# ============================================================

# Evaluator: checks correctness against reference using LLM-as-judge
eval_llm = LLM

CORRECTNESS_GRADER_PROMPT = (
  "You are a grader. Given a question, a submission, and a reference answer, "
  "determine if the submission is correct, accurate, and factual compared to "
  "the reference answer.\n\n"
  "Question: {question}\n"
  "Submission: {submission}\n"
  "Reference: {reference}\n\n"
  "Respond with ONLY 'Y' if correct or 'N' if incorrect."
)

HELPFULNESS_GRADER_PROMPT = (
  "You are a grader. Given a question and a response, "
  "determine if the response is helpful, clear, and easy to understand.\n\n"
  "Question: {question}\n"
  "Response: {response}\n\n"
  "Respond with ONLY 'Y' if helpful or 'N' if not helpful."
)


# correctness evaluator
def correctness(run, example) -> dict:
  """LLM-as-judge evaluator for correctness against reference answer."""
  prediction = run.outputs.get("answer", "")
  reference = example.outputs.get("answer", "")
  question = example.inputs.get("question", "")

  grade_prompt = ChatPromptTemplate.from_template(CORRECTNESS_GRADER_PROMPT)
  result = eval_llm.invoke(
    grade_prompt.format(question=question, submission=prediction, reference=reference)
  )
  score = 1.0 if result.content.strip().upper() == "Y" else 0.0
  return {"key": "correctness", "score": score}


# helpfulness evaluator
def helpfulness(run, example) -> dict:
  """LLM-as-judge evaluator for helpfulness (no reference needed)."""
  prediction = run.outputs.get("answer", "")
  question = example.inputs.get("question", "")

  grade_prompt = ChatPromptTemplate.from_template(HELPFULNESS_GRADER_PROMPT)
  result = eval_llm.invoke(grade_prompt.format(question=question, response=prediction))
  score = 1.0 if result.content.strip().upper() == "Y" else 0.0
  return {"key": "helpfulness", "score": score}


# contains_answer evaluator
def contains_answer(run, example) -> dict:
  """
  Custom evaluator — checks if the response contains
  key terms from the expected answer.
  """
  prediction = run.outputs.get("answer", "").lower()
  reference = example.outputs.get("answer", "").lower()

  # Extract key words from reference (words > 3 chars)
  key_words = [word for word in reference.split() if len(word) > 3]

  # Check if at least 50% of key words appear in prediction
  if not key_words:
    return {"key": "contains_answer", "score": 1.0}

  matches = sum(1 for word in key_words if word in prediction)
  score = matches / len(key_words)

  return {"key": "contains_answer", "score": score}


# ============================================================
# Step 2: Run the evaluation
# ============================================================


def run_evaluation(dataset_name: str):
  """Run evaluation against the dataset."""

  results = evaluate(
    qa_target_v1,
    data=dataset_name,
    evaluators=[correctness, helpfulness, contains_answer],
    experiment_prefix="qa-chain-v1",  # Tags this run for comparison
    max_concurrency=2,
  )

  # Print Results in a readable format
  print_results(results, "v1")

  return results


# ============================================================
# Step 3: Compare experiments (after model or prompt change)
# ============================================================


def run_comparison(dataset_name: str):
  """
  Run a second experiment with a different config,
  then compare in LangSmith dashboard.
  """

  # New prompt — more detailed instructions
  detailed_prompt_template = (
    "Answer this question accurately and concisely. "
    "If it's a factual question, be precise. "
    "If it's a math question, show just the answer.\n\n"
    "Question: {question}"
  )
  detailed_prompt = ChatPromptTemplate.from_template(detailed_prompt_template)
  v2_chain = detailed_prompt | LLM

  # qa_target_v2 function for LangSmith evaluation. Must accept a dict (inputs) and return a dict (outputs).
  @traceable(name="qa_target_v2")
  def qa_target_v2(inputs: dict) -> dict:
    response = v2_chain.invoke({"question": inputs["question"]})
    return {"answer": response.content}

  # print("\nRunning v2 experiment for comparison...\n")

  results = evaluate(
    qa_target_v2,
    data=dataset_name,
    evaluators=[correctness, helpfulness, contains_answer],
    experiment_prefix="qa-chain-v2",  # Different prefix for comparison
    max_concurrency=2,
  )

  print_results(results, "v2")

  print("\n\033[92mDone! Compare v1 vs v2 in LangSmith dashboard:\033[0m")
  print("  → Go to your LangSmith project → Datasets → qa-eval-dataset")
  print("  → Click 'Compare Experiments' to see v1 vs v2 side by side")

  return results


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
  # Step 1: Create dataset
  print_section("STEP 1: Create LangSmith Dataset - qa-eval-dataset")
  dataset_name = create_eval_dataset()

  # Step 2: Run first evaluation (v1)
  print_section(f"STEP 2: Run First Evaluation (v1) - {dataset_name}")
  run_evaluation(dataset_name)

  # Step 3: Run second evaluation (v2) for comparison
  print_section(f"STEP 3: Run Second Evaluation (v2) - {dataset_name} Comparison")
  run_comparison(dataset_name)

  print("\033[95mAll experiments logged to LangSmith!\033[0m\n")
