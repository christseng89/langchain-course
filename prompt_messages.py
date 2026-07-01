from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def section(title):
  print(f"\n{CYAN}{'─' * 60}\n  {title}\n{'─' * 60}{RESET}")


def print_messages(messages):
  for msg in messages:
    print(f"{type(msg).__name__}: {msg.content}")


# ChatPromptTemplate from_template
prompt = ChatPromptTemplate.from_template("Tell me a `{adjective}` joke about `{topic}`.")
# format and inspect
messages = prompt.format_messages(adjective="funny", topic="chickens")
section("ChatPromptTemplate — single template")
print_messages(messages)

# multi-message templates ChatPromptTemplate from_messages
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "You are a helpful assistant that translates `{input_language}` to `{output_language}`.",
    ),
    ("human", "Translate the following text: `{text}`"),
  ]
)

messages = prompt.format_messages(
  input_language="English", output_language="French", text="I love programming."
)

section("ChatPromptTemplate — multi-message template")
print_messages(messages)

model = init_chat_model(model="gpt-4o-mini", temperature=0)
response = model.invoke(messages)

section("Model response — translation")
print(response.content)

# Chain of prompt and model
chain = prompt | model
response = chain.invoke(
  {"input_language": "English", "output_language": "French", "text": "I love programming."}
)

section("Chain Model response — translation")
print(response.content)

# Message Types: HumanMessage, AIMessage, SystemMessage, ToolMessage, ChatMessage
from langchain_core.messages import (
  AIMessage,
  ChatMessage,
  FunctionMessage,
  ToolMessage,
)

messages = [
  HumanMessage(content="Hello!"),
  AIMessage(content="Hi there! How can I assist you today?"),
  SystemMessage(content="This is a system message."),
  ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
  ChatMessage(role="user", content="This is a general chat message."),
  FunctionMessage(content="This is a function message.", name="my_function"),
]

section("Message Types — HumanMessage, AIMessage, SystemMessage, ToolMessage, ChatMessage")
# print(messages)
print_messages(messages)

# Fewshot ChatMessagePromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate

example_prompt = ChatPromptTemplate.from_messages(
  [
    ("human", "{input}"),
    ("ai", "{output}"),
  ]
)

examples = [
  {"input": "happy", "output": "sad"},
  {"input": "tall", "output": "short"},
]

fewshot_prompt = FewShotChatMessagePromptTemplate(
  example_prompt=example_prompt,
  examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Give the opposite of each word."),
    fewshot_prompt,
    ("human", "{input}"),
  ]
)

messages = final_prompt.format_messages(input="test")
section("FewShotChatMessagePromptTemplate — Test Single example")
print_messages(messages)

antonym_inputs = [
  "fast",
  "bright",
  "hot",
  "happy",
  "clever",
  "responsible",
  "easy",
  "agree",
  "verify",
  "accept",
  "promote",
]

model = init_chat_model(model="gpt-4o-mini", temperature=0)
section("FewShotChatMessagePromptTemplate — AI antonym response")
for word in antonym_inputs:
  messages = final_prompt.format_messages(input=word)
  # print_messages(messages)
  print(f"Invoking MODEL for input: {YELLOW}{word}{RESET}")
  response = model.invoke(messages)
  print(f"{YELLOW}{word} → {response.content}{RESET}\n")

  print(f"Invoking CHAIN for input: {GREEN}{word}{RESET}")
  chain = final_prompt | model
  response = chain.invoke({"input": word})
  print(f"{GREEN}{word} → {response.content}{RESET}\n")

# Reusable components
system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])
user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

# Combine
prompt = system_prompt + user_prompt
messages = prompt.format_messages(role="helpful assistant", question="What is AI?")
section("Reusable components — combined system + user prompt")
print_messages(messages)

chain = prompt | model
response = chain.invoke({"role": "helpful assistant", "question": "What is AI?"})
print(f"\nAI Model Response:\n{response.content[:100]}\n")
