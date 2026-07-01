"""
LangChain Core Concepts - LCEL and Runnables
"""

import inspect

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()


YELLOW = "\033[93m"
RESET = "\033[0m"


def print_result(input_val, prompt_val, output_val, description=None, stream=False):
  if description:
    print(f"{YELLOW}{inspect.cleandoc(description)}{RESET}")
  print(f"Input:  {input_val}")
  print(f"Prompt: {prompt_val}")
  if stream:
    print("Output: ", end="", flush=True)
    for chunk in output_val:
      print(chunk, end="", flush=True)
    print("\n")
  else:
    print(f"Output: {output_val}")
    print("\n")


def demo_basic_chain():
  """Demonstrates a basic chain using LCEL and Runnables."""

  # Component 1: Define the prompt template using LCEL
  prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer in one sentence: {question}"
  )

  # Compose with pipe operator
  chain = prompt | model | parser

  input_data = {"question": "What is LangChain?"}
  result = chain.invoke(input_data)
  print_result(
    input_data["question"],
    f"You are a helpful assistant. Answer in one sentence: {input_data['question']}",
    result,
    description=demo_basic_chain.__doc__,
  )

  return chain


def demo_batch_exectution():
  """Demonstrate batch execution for multiple inputs."""
  prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

  chain = prompt | model | parser

  # Batch - run with multiple inputs
  inputs = [
    {"text": "Hello, how are you?"},
    {"text": "What is your name?"},
    {"text": "Where is the nearest restaurant?"},
  ]
  results = chain.batch(inputs)

  first_input, first_output = inputs[0], results[0]
  print_result(
    first_input["text"],
    f"Translate to French: {first_input['text']}",
    first_output,
    description=demo_batch_exectution.__doc__,
  )


def demo_streaming():
  """Demonstrate streaming for real-time output."""
  prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

  chain = prompt | model | parser

  topic = "nature"
  print_result(
    topic,
    f"Write a haiku about: {topic}",
    chain.stream({"topic": topic}),
    description=demo_streaming.__doc__,
    stream=True,
  )


def demo_schema_inspection():
  """Demonstrate input/output schema inspection."""
  prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")

  chain = prompt | model | parser

  input_schema = chain.input_schema.model_json_schema()
  output_schema = chain.output_schema.model_json_schema()

  print_result(
    input_schema,
    "Summarize the following text: <text>",
    output_schema,
    description=demo_schema_inspection.__doc__,
  )


# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
  """
  EXERCISE: Create a chain that:
  1. Takes a product name and target audience
  2. Generates a marketing tagline
  3. Returns just the tagline as a string

  Test with: product="AI Course", audience="developers"
  """

  # YOUR CODE HERE
  prompt = ChatPromptTemplate.from_template(
    "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
  )
  chain = prompt | model | parser

  input_data = {"product": "AI Course", "audience": "developers"}
  result = chain.invoke(input_data)
  print_result(
    input_data,
    f"Create a marketing tagline for a product named '{input_data['product']}' targeting '{input_data['audience']}'.",
    result,
    description=exercise_first_chain.__doc__,
  )


def new_way():
  # the univeral way to initialize a model
  model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)

  # Or provider-specific (still works)

  from langchain_openai import ChatOpenAI

  openai_model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.7, max_tokens=1500, timeout=30, max_retries=3
  )

  from langchain_anthropic import ChatAnthropic

  anthropic_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")


if __name__ == "__main__":
  demo_basic_chain()
  demo_batch_exectution()
  demo_streaming()
  demo_schema_inspection()
  exercise_first_chain()
