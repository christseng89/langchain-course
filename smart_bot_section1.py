"""
Section 1 Project: Smart Q&A Bot by Using LangSmith Tracing
A production-ready question-answering bot with structured output
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

load_dotenv()


def print_description(text):
  print(f"\n\033[94m{'***'} {text} {'***'}\033[0m\n")


def print_section(title, newline_before=False):
  prefix = "\n" if newline_before else ""
  print(f"{prefix}\033[92m{'=' * 60}\033[0m")
  print(f"\033[92m{title}\033[0m")
  print(f"\033[92m{'=' * 60}\033[0m")


# -- LangSmith Configuration --
if os.getenv("LANGSMITH_API_KEY"):
  print("LangSmith API key found. Tracing is enabled.")
  os.environ["LANGSMITH_TRACING"] = "true"
  os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot Project")
  print(f"LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")


messages = [
  (
    "system",
    """You are a knowledgeable Q&A assistant.

Your guidelines:
- Answer questions accurately and concisely
- Be honest about uncertainty - set confidence to 'low' if unsure
- Provide clear reasoning for your answers
- Suggest relevant follow-up questions
- Indicate if external sources would help

Always respond with accurate, helpful information.""",
  ),
  ("human", "{question}"),
]

questions = [
  "What is the capital of France?",
  "Explain the theory of relativity in Chinese.",
  "How does photosynthesis work in Chinese?",
  "What is the characteristic of a Russian Blue cat in Chinese?",
]

batch_questions = [
  "What is Python?",
  "What is JavaScript?",
  "What is Rust?",
  "What is Go?",
  "What is Java?",
]


# Schema Definition
class QAResponse(BaseModel):
  answer: str = Field(description="The answer to the user's question.")
  confidence: str = Field(description="Confidence level: high, medium, or low")
  reasoning: str = Field(description="The reasoning behind the answer provided.")
  follow_up_questions: List[str] = Field(
    description="A list of follow-up questions related to the topic.",
    default_factory=list,
  )
  sources_needed: bool = Field(
    description="Indicates whether sources are needed for the answer.",
    default=False,
  )

  # Bot implementation


class SmartQABot:
  def __init__(
    self,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
  ):
    self.model = ChatOpenAI(
      model=model_name,
      temperature=temperature,
    ).with_structured_output(QAResponse)
    self.prompt = ChatPromptTemplate.from_messages(messages)
    self.chain = self.prompt | self.model

  @traceable(name="ask_question", run_type="chain")
  def ask_question(self, question: str) -> QAResponse:
    try:
      response = self.chain.invoke({"question": question})
      return response
    except Exception as e:
      # return a greaceful error response
      return QAResponse(
        answer="I'm sorry, I couldn't process your question at this time.",
        confidence="low",
        reasoning=str(e),
        follow_up_questions=["Could you please try again later?"],
        sources_needed=True,
      )

  @traceable(name="ask_batch", run_type="chain")
  def ask_batch(self, questions: List[str]) -> List[QAResponse]:
    """Ask multiple questions in parallel."""
    inputs = [{"question": q} for q in questions]
    return self.chain.batch(inputs)


# Demo Usage
def demo_qa_bot():
  bot = SmartQABot()
  for question in questions:
    response = bot.ask_question(question)

    print_section(f"Question: {question}")

    print(f"Answer: {response.answer}\n")
    print(f"Confidence: {response.confidence}\n")
    print(f"Reasoning: {response.reasoning}\n")
    print(f"Follow-up Questions: {response.follow_up_questions}\n")
    print(f"Sources Needed: {response.sources_needed}\n")


@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
  """Demonstrate batch processing."""

  bot = SmartQABot()
  print_section("BATCH PROCESSING DEMO")
  responses = bot.ask_batch(batch_questions)

  for q, r in zip(batch_questions, responses):
    print(f"\n{q}")
    print(f"  {r.answer}")
    print(f"  Confidence: {r.confidence}")


@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
  """Demonstrate error handling."""

  bot = SmartQABot()
  print_section("ERROR HANDLING DEMO")
  # Test with a very long question (edge case)
  long_question = "What is " + "very " * 1000000 + "important?"

  response = bot.ask_question(long_question)
  print(f"\nQuestion: {long_question[:100]}...\n")
  print(f"Answer: {response.answer}\n")
  print(f"Confidence: {response.confidence}")


if __name__ == "__main__":
  try:
    print_description(
      "Smart Q&A Bot — structured output with confidence, reasoning, and follow-up questions"
    )
    demo_qa_bot()

    print_description("Batch Processing — ask multiple questions in parallel")
    demo_batch_processing()

    print_description("Error Handling — graceful degradation on edge cases")
    demo_error_handling()

    print_description("Section 1 Complete!")
    print("""
What you learned:
- LangChain ecosystem overview
- Environment setup with uv
- Core concepts: Runnables, LCEL, pipe operator
- Working with multiple LLM providers
- Prompt templates and message types
- Output parsers and structured output
- Building a production Q&A bot
- LangSmith tracing with @traceable decorator

Next: Section 2 - Chains, RAG & Memory
        """)
  finally:
    pass
  # uncomment the line below to flush traces to LangSmith, but you'll alse see an error at the end of a run, which is not harmful at all, but annoying!
  # Client().flush()  # Ensure all traces are sent to LangSmith
