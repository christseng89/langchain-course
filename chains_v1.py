"""
Understanding Chains in LangChain V.1
LCEL patterns, composition, and debugging
"""

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
  RunnableBranch,
  RunnableLambda,
  RunnableParallel,
  RunnablePassthrough,
)
from langsmith import traceable

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

model = init_chat_model(model="gpt-4o-mini", temperature=0)


def print_section(title):
  """Print a section header in blue."""
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


# Basic chain example: demonstrate Runnable composition
@traceable(name="demo_basic_chain", run_type="chain")
def demo_basic_chain():
  prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
  parser = StrOutputParser()
  chain = prompt | model | parser

  result = chain.invoke(
    {"text": "LangChain is a framework for developing applications powered by language models."}
  )
  print(f"Summary: {result}  ")


# Parallel chain example: demonstrate RunnableParallel
@traceable(name="demo_parallel_chain", run_type="chain")
def demo_parallel_chain():
  """Run multiple chains in parallel."""
  # define individual chains
  summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
  keywords_prompt = ChatPromptTemplate.from_template(
    "Extract 5 keywords in the following text: {text}\nReturn as a comma-separated list."
  )
  sentiment_prompt = ChatPromptTemplate.from_template(
    "What is the sentiment of the following text? {text}"
  )

  parser = StrOutputParser()

  # Parallel chain execution
  chain = RunnableParallel(
    summary=summarize_prompt | model | parser,
    keywords=keywords_prompt | model | parser,
    sentiment=sentiment_prompt | model | parser,
  )

  text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """

  results = chain.invoke({"text": text})
  print("Parallel Chain Analysis Results:\n")
  print(f"- Summary: {results['summary']}\n")
  print(f"- Keywords: {results['keywords']}\n")
  print(f"- Sentiment: {results['sentiment']}\n")


# Passthrough chain example: demonstrate RunnablePassthrough vs itemgetter
@traceable(name="demo_passthrough_chain", run_type="chain")
def demo_passthrough_chain():
  """A chain that demonstrates passthrough functionality, comparing
  RunnablePassthrough (needs an unwrap step) vs itemgetter (doesn't)."""
  prompt = ChatPromptTemplate.from_template(
    "Original question: {question}\nContext: {context}\n\nAnswer the question based on the context."
  )

  # simulate a retrieve operation
  def fake_retriever(input_dict):
    return " LangChain was created by Harrison Chase in 2022."

  chain = (
    RunnableParallel(context=RunnableLambda(fake_retriever), question=RunnablePassthrough())
    | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]["question"]})
    | prompt
    | model
    | StrOutputParser()
  )

  result = chain.invoke({"question": "Who created LangChain?"})
  print(f"Answer: {result}")

  # Easier way: use itemgetter to extract the question from the input dict
  print_section("Passthrough Chain (itemgetter)")

  chain = (
    RunnableParallel(context=RunnableLambda(fake_retriever), question=itemgetter("question"))
    | prompt
    | model
    | StrOutputParser()
  )

  result = chain.invoke({"question": "Who created LangChain?"})
  print(f"Answer: {result}")


# Parallel chain branching example: demonstrate RunnableBranch
@traceable(name="demo_parallel_chain_branching", run_type="chain")
def demo_parallel_chain_branching():
  """A chain that demonstrates branching functionality."""

  # Different prompts for different intents
  code_prompt = ChatPromptTemplate.from_template("You are a coding expert. Help with: {input}")
  general_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer: {input}")

  # Classifier
  classifier_prompt = ChatPromptTemplate.from_template(
    "Classify this as 'code' or 'general': {input}\nReturn only the classification."
  )
  classifer_chain = classifier_prompt | model | StrOutputParser()

  # Branching chain  based on classification
  def is_code_question(input_dict):
    classification = classifer_chain.invoke(input_dict)
    print(f"\nClassification: {classification}")
    print(f"Input: {input_dict['input']}")
    print(f"Is code question? {'code' in classification.lower()}\n")
    return "code" in classification.lower()

  branch_chain = RunnableBranch(
    (is_code_question, code_prompt | model | StrOutputParser()),
    general_prompt | model | StrOutputParser(),  # default branch
  )

  # Test
  questions = [
    "How do I write a for loop in Python?",
    "What's the weather like today?",
    "Can you help me debug this JavaScript code?",
    "What is the capital of France?",
  ]

  for question in questions:
    result = branch_chain.invoke({"input": question})
    print(f"Q: {question}")
    print(f"A: {result[:100]}...\n")


# Debugging example: demonstrate how to inspect intermediate steps
@traceable(name="demo_debugging", run_type="chain")
def demo_debugging():
  # prompt = ChatPromptTemplate.from_template("Say hello to {name}")
  prompt = ChatPromptTemplate.from_template("Say hello to {title} {name}")
  chain = prompt | model | StrOutputParser()

  # Method 1: Get configuration
  print("\n\033[93m1: Get configuration:\033[0m")
  print("Chain input schema:", chain.input_schema.model_json_schema())
  print("Chain output schema:", chain.output_schema.model_json_schema())

  # Method 2: Use with_config for tacing
  print("\n\033[93m2: Use with_config for tracing:\033[0m")
  result = chain.with_config(
    run_name="greeting_chain",
    tags=["demo debugging"],
  ).invoke({"name": "Alice", "title": "Ms."})
  print(f"Greeting: {result}")

  # Method 3: Inspect intermediate steps
  # Using RunnableLambda for logging
  print("\n\033[93m3: Inspect intermediate steps:\033[0m")
  prompt = ChatPromptTemplate.from_template("Say hello to {name}")

  def log_step(x, step_name=""):
    if hasattr(x, "content"):
      content = x.content
    elif hasattr(x, "to_string"):  # e.g. ChatPromptValue
      content = x.to_string()
    else:
      content = str(x)
    print(f"[{step_name}] {type(x).__name__}\nContent: {content[:100]}\n")
    return x

  debug_chain = (
    prompt
    | RunnableLambda(lambda x: log_step(x, "after_prompt"))
    | model
    | RunnableLambda(lambda x: log_step(x, "after_model"))
    | StrOutputParser()
  )

  print("\033[93mDebug chain execution:\033[0m")
  result = debug_chain.invoke({"name": "Debug"})
  print(f"\033[92mGreeting: {result}\033[0m\n")

  result = debug_chain.invoke({"name": "Chris"})
  print(f"\033[92mGreeting: {result}\033[0m\n")

  result = debug_chain.invoke({"name": "LLM"})
  print(f"\033[92mGreeting: {result}\033[0m\n")


if __name__ == "__main__":
  print_section("Basic Chain")
  demo_basic_chain()

  print_section("Parallel Chain")
  demo_parallel_chain()

  print_section("Passthrough Chain")
  demo_passthrough_chain()

  print_section("Parallel Chain Branching via Classification")
  demo_parallel_chain_branching()

  print_section("Debugging")
  demo_debugging()
