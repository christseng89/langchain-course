"""
Conversation Memory in LangChain
Modern approaches to maintaining conversation context
"""

import os
import sqlite3
from typing import Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import (
  BaseChatMessageHistory,
  InMemoryChatMessageHistory,
)
from langchain_core.messages import (
  AIMessage,
  HumanMessage,
  SystemMessage,
  trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


# Shared prompt with history placeholder, reused across every demo/exercise below
PROMPT = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant. Remember user details."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
  ]
)

LLM = init_chat_model("gpt-4o-mini")
LLM1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def get_session_history(
  session_id: str, store: Dict[str, InMemoryChatMessageHistory]
) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]


# Basic Conversation Memory with RunnableWithMessageHistory
def demo_basic_memory():
  """Basic conversation memory with RunnableWithMessageHistory."""

  chain = PROMPT | LLM1 | StrOutputParser()
  # Session storage
  store: Dict[str, InMemoryChatMessageHistory] = {}
  session_id = "user_123"
  # Wrap with history
  chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda sid: get_session_history(sid, store),
    input_messages_key="input",
    history_messages_key="history",
  )

  # Configuration for this session
  config = {"configurable": {"session_id": session_id}}

  # Conversation
  messages = [
    "Hi! My name is Paulo.",
    "I'm learning about LangChain.",
    "What's my name and what am I learning?",
  ]

  print("\nConversation:")
  for msg in messages:
    print(f"\n\033[92mUser: {msg}\033[0m")
    response = chain_with_history.invoke({"input": msg}, config=config)
    print(f"AI: {response}")

  # Show stored history
  print(
    f"\n\033[92m--- InMemory Messages History ({len(store[session_id].messages)} messages) ---\033[0m"
  )
  for msg in store[session_id].messages:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  {role}: {msg.content[:50]}...")


# Multiple Conversation Sessions
def demo_multi_sessions():
  """Simulate multiple users with separate conversation histories."""

  chain = PROMPT | LLM1 | StrOutputParser()
  store: Dict[str, InMemoryChatMessageHistory] = {}

  chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda sid: get_session_history(sid, store),
    input_messages_key="input",
    history_messages_key="history",
  )

  # Simulate two users
  user_a_config = {"configurable": {"session_id": "user_a"}}
  user_b_config = {"configurable": {"session_id": "user_b"}}

  # User A conversation
  print("\n--- User A ---")
  message_a = "My name is John and my favorite language is Python"
  print(f"\033[92mUser A: {message_a}\033[0m")
  resp = chain_with_history.invoke({"input": message_a}, config=user_a_config)
  print(f"AI: {resp}")

  # User B conversation
  print("\n--- User B ---")
  message_b = "My name is Jane and I love JavaScript"
  print(f"\033[93mUser B: {message_b}\033[0m")
  resp = chain_with_history.invoke({"input": message_b}, config=user_b_config)
  print(f"AI: {resp}")

  # Ask each user about their preference
  print("\n--- Asking each about their preference ---")

  question = "What's my name and what's my favorite language?"
  print(f"\n\033[92mUser A: {question}\033[0m")
  resp = chain_with_history.invoke({"input": question}, config=user_a_config)
  print(f"AI: {resp}")

  print(f"\n\033[93mUser B: {question}\033[0m")
  resp = chain_with_history.invoke({"input": question}, config=user_b_config)
  print(f"AI: {resp}")


# Demonstrate message trimming to fit within context window
def demo_message_trimming():
  """Trim messages to fit context window."""

  # Simulate a long conversation
  messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="What is Python?"),
    AIMessage(
      content="Python is a high-level programming language known for readability and versatility. It's used in web development, data science, AI, and automation."
    ),
    HumanMessage(content="How do I install it?"),
    AIMessage(
      content="You can install Python from python.org or use package managers like apt, brew, or chocolatey. I recommend Python 3.12+ for new projects."
    ),
    HumanMessage(content="What about pip?"),
    AIMessage(
      content="Pip is Python's package installer. It comes with Python 3.4+. Use 'pip install package_name' to install packages. Consider using virtual environments with venv or uv."
    ),
    HumanMessage(content="Can you summarize everything we discussed?"),
  ]

  print(f"\033[92m\nOriginal: {len(messages)} messages\033[0m")

  # Trim to last N tokens
  trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",
    token_counter=LLM,
    include_system=True,  # Always keep system message
    allow_partial=False,
  )

  print(f"\033[93mAfter trimming (max 100 tokens): {len(trimmed)} messages\033[0m")
  # print("\nTrimmed messages:")
  for msg in trimmed:
    role = type(msg).__name__.replace("Message", "")
    print(f"  {role}: {msg.content}")

  # Send the trimmed messages to the LLM to get the actual answer
  response = LLM.invoke(trimmed)
  print(f"\n\033[92mAI Response from trimmed context:\033[0m \n{response.content}")


# Demonstrate sliding window memory (keep last K exchanges)
def demo_windowed_memory():
  """Implement sliding window memory manually."""

  MAX_RECENT = 2

  class WindowedChatHistory(InMemoryChatMessageHistory):
    """Chat history that keeps only last k message pairs."""

    k: int = 3  # Pydantic field - number of exchange pairs to keep

    def add_messages(self, messages):
      super().add_messages(messages)
      # Keep only last k pairs (2k messages: human + ai)
      if len(self.messages) > self.k * 2:
        self.messages = self.messages[-(self.k * 2) :]

  store: Dict[str, WindowedChatHistory] = {}

  def get_windows_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
      store[session_id] = WindowedChatHistory(k=MAX_RECENT)
    return store[session_id]

  chain = PROMPT | LLM1 | StrOutputParser()
  chain_with_history = RunnableWithMessageHistory(
    chain,
    get_windows_history,
    input_messages_key="input",
    history_messages_key="history",
  )

  config = {"configurable": {"session_id": "windowed_test"}}

  # Simulate a conversation with more than 2 pairs
  exchanges = [
    "My name is Paulo",
    "I live in Seattle",
    "I work as an AI engineer",
    "I have 2 cats",
    "What do you remember about me?",
    "I'm 32 years old male",
  ]

  # print("\nConversation with k=2 window:")
  for i, msg in enumerate(exchanges, 1):
    print(f"\033[92m\nUser: {msg}\033[0m")
    response = chain_with_history.invoke({"input": msg}, config=config)
    print(f"\033[93mAI Response: \n{response}\033[0m")

    # Show window state after each exchange so students SEE it sliding
    history = store["windowed_test"].messages
    print(f"  [Window: {len(history)} msgs] ", end="")
    facts_in_memory = [m.content[:40] for m in history if isinstance(m, HumanMessage)]
    print(f"Remembers: {facts_in_memory}")

  # Final state - show what survived and what was lost
  print(
    f"\033[38;5;208m\nRESULT: Window only kept last `{MAX_RECENT}` messages, due to WindowedChatHistory(`k={MAX_RECENT}`)!\033[0m"
  )


# Demo Summary Memory
def demo_summary_memory():
  """End-to-end summary memory: auto-summarize old messages, keep recent ones verbatim."""
  # Chat Prompt, Chat Chain (LLM1)
  chat_prompt = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "You are a helpful assistant. Be concise.\n\nSummary of earlier conversation:\n{summary}",
      ),
      MessagesPlaceholder(variable_name="recent_messages"),
      ("human", "{input}"),
    ]
  )
  chat_chain = chat_prompt | LLM1 | StrOutputParser()

  # Summary Prompt, LLM, Chain
  summarize_prompt = ChatPromptTemplate.from_template(
    "Condense the current summary and new messages into a single updated summary "
    "(2-3 sentences). Preserve all key facts about the user.\n\n"
    "Current summary:\n{current_summary}\n\n"
    "New messages:\n{new_messages}\n\n"
    "Updated summary:"
  )
  summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  summarize_chain = summarize_prompt | summary_llm | StrOutputParser()

  # --- State ---
  running_summary = ""  # starts empty
  recent_messages = []  # full message objects
  MAX_RECENT = 4  # keep last 4 messages (2 exchanges) before summarizing

  # --- Conversation ---
  user_inputs = [
    "My name is Paulo and I'm from Seattle",
    "I work as an AI engineer building RAG systems",
    "I have 2 cats named Luna and Milo",
    "I'm building a LangChain course for Udemy",
    "I'm 32 years old male",
    "I'm going to work from 8:30am for 8 hours every workday",
    "What do you know about me? List everything.",
  ]

  print(f"\033[38;5;208mConfig: keep last {MAX_RECENT} messages, summarize the rest\n\033[0m")

  for user_input in user_inputs:
    print(f"\033[92mUser: {user_input}\033[0m\n")

    # 1. Call the LLM with summary + recent messages + new input
    response = chat_chain.invoke(
      {
        "summary": (running_summary if running_summary else "No prior conversation."),
        "recent_messages": recent_messages,
        "input": user_input,
      }
    )
    print(f"\033[93mAI Response: \n{response}\n\033[0m")

    # 2. Add this user_input to recent messages
    recent_messages.append(HumanMessage(content=user_input))
    recent_messages.append(AIMessage(content=response))

    # 3. If recent messages exceed limit, summarize the oldest ones
    if len(recent_messages) > MAX_RECENT:  # One for User, one for AI
      # Take the oldest messages that will be summarized away
      messages_to_summarize = recent_messages[:-MAX_RECENT]
      new_messages = "\n".join(
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages_to_summarize
      )
      # Update the running summary
      running_summary = summarize_chain.invoke(
        {
          "current_summary": (running_summary if running_summary else "None yet."),
          "new_messages": new_messages,
        }
      )
      # Keep only the most recent messages
      print("\033[38;5;208mSUMMARIZED...\033[0m")
      recent_messages = recent_messages[-MAX_RECENT:]
      print(f" >>> Compressed `{len(messages_to_summarize)}` old messages")
      print(f" >>> Running Summary: {running_summary}")

  # --- Final state ---

  print("\033[38;5;208m\nFINAL MEMORY STATE\033[0m")

  print(f"\nRunning Summary (compressed old context):\n  {running_summary}")
  print(f"\nRecent Messages Length: ({len(recent_messages)}):")


# Exercise
def exercise_persistent_memory():
  """
  EXERCISE: Build a chatbot with:
  1. Persistent memory (SQLite)
  2. Automatic summarization after 10 messages
  3. User preference tracking

  Hint: Combine RunnableWithMessageHistory with SQLChatMessageHistory
  """

  print("=" * 60)
  print("EXERCISE: Persistent Memory Chatbot")
  print("=" * 60)

  from langchain_community.chat_message_histories import SQLChatMessageHistory
  from sqlalchemy import create_engine

  # Use SQLite for persistence
  DB_PATH = "./chat_history.db"

  # Single shared engine so we can dispose it (closes pooled connections)
  # before deleting the file -- SQLChatMessageHistory(connection=<str>)
  # creates a brand-new engine per call, and those are never released,
  # which keeps the file locked on Windows.
  engine = create_engine(f"sqlite:///{DB_PATH}")

  def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection=engine)

  chain = PROMPT | LLM1 | StrOutputParser()

  chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
  )

  config = {"configurable": {"session_id": "persistent_user"}}

  print("\nPersistent memory chatbot:")
  print("(Messages saved to SQLite database)\n")

  # Test conversation
  test_messages = [
    "Remember that I prefer dark mode themes",
    "What theme do I prefer?",
  ]

  for msg in test_messages:
    print(f"User: {msg}")
    response = chain_with_history.invoke({"input": msg}, config=config)
    print(f"AI: {response}\n")

  print(f"Database created: {DB_PATH}")
  print("Messages persist across restarts!")

  # Cleanup for demo -- dispose the engine first to release pooled
  # connections, otherwise Windows keeps the file locked.
  engine.dispose()
  if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def exercise_persistent_memory_proof():
  """
  EXERCISE: Build a chatbot with:
  1. Persistent memory (SQLite)
  2. Proof that messages survive across separate chain instances
  3. User preference tracking

  Key idea: We create the chain TWICE to simulate two separate program runs.
  The second run reads from the same SQLite DB and recalls what the first run stored.
  """

  print("=" * 60)
  print("EXERCISE: Persistent Memory Chatbot")
  print("=" * 60)

  from langchain_community.chat_message_histories import SQLChatMessageHistory
  from sqlalchemy import create_engine

  DB_PATH = "./chat_history.db"
  CONNECTION_STRING = f"sqlite:///{DB_PATH}"
  SESSION_ID = "persistent_user"

  # Clean slate
  if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

  # Single shared engine so we can dispose it (closes pooled connections)
  # before deleting the file -- SQLChatMessageHistory(connection=<str>)
  # creates a brand-new engine per call, and those are never released,
  # which keeps the file locked on Windows.
  engine = create_engine(CONNECTION_STRING)

  # --- Helper: build a fresh chain (simulates a new program run) ---
  def build_chain():
    def get_session_history(sid: str) -> BaseChatMessageHistory:
      return SQLChatMessageHistory(
        session_id=sid,
        connection=engine,
      )

    chain = PROMPT | LLM1 | StrOutputParser()

    return RunnableWithMessageHistory(
      chain,
      get_session_history,
      input_messages_key="input",
      history_messages_key="history",
    )

  config = {"configurable": {"session_id": SESSION_ID}}

  # =====================================================
  # RUN 1 -- Store preferences (simulates first session)
  # =====================================================
  print("\n--- RUN 1: Storing preferences ---\n")

  chain_v1 = build_chain()

  run1_messages = [
    "My name is Paulo. I prefer dark mode themes and Python over JavaScript.",
    "I also like my responses concise -- no fluff.",
  ]

  for msg in run1_messages:
    print(f"User: {msg}")
    response = chain_v1.invoke({"input": msg}, config=config)
    print(f"AI:   {response}\n")

  # Throw away the chain object entirely -- no in-memory state survives
  del chain_v1

  # =====================================================
  # PROOF: Inspect the raw SQLite database
  # =====================================================
  print("--- DATABASE PROOF ---\n")
  print(f"Database file exists: {os.path.exists(DB_PATH)}")
  print(f"Database size: {os.path.getsize(DB_PATH)} bytes\n")

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.execute("SELECT * FROM message_store ORDER BY rowid")
  rows = cursor.fetchall()
  print(f"Total messages stored in DB: {len(rows)}\n")

  for i, row in enumerate(rows):
    print(
      f"  Row {i + 1}: session={row[0] if len(row) > 0 else 'N/A'}, "
      f"message (first 80 chars): {str(row[1])[:80] if len(row) > 1 else 'N/A'}..."
    )
  conn.close()

  # =====================================================
  # RUN 2 -- Brand new chain, same DB (simulates restart)
  # =====================================================
  print("\n--- RUN 2: Fresh chain, testing recall ---\n")

  chain_v2 = build_chain()

  recall_questions = [
    "What's my name?",
    "What theme do I prefer?",
    "What programming language do I prefer?",
    "How do I like my responses?",
  ]

  for msg in recall_questions:
    print(f"User: {msg}")
    response = chain_v2.invoke({"input": msg}, config=config)
    print(f"AI:   {response}\n")

  del chain_v2

  # =====================================================
  # FINAL: Show total messages accumulated
  # =====================================================
  print("--- FINAL DATABASE STATE ---\n")
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.execute("SELECT COUNT(*) FROM message_store")
  count = cursor.fetchone()[0]
  conn.close()

  print(f"Total messages in DB after both runs: {count}")
  print("Key insight: The second chain had ZERO in-memory history.")
  print("Everything was loaded from SQLite -- true persistence!")

  # Cleanup -- dispose the engine first to release pooled connections,
  # otherwise Windows keeps the file locked and os.remove raises WinError 32.
  engine.dispose()
  if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def _print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


if __name__ == "__main__":
  # _print_section("Demo Basic Memory")
  # demo_basic_memory()

  # _print_section("Demo Multi-Sessions")
  # demo_multi_sessions()

  # _print_section("Demo Message Trimming")
  # demo_message_trimming()

  # _print_section("Demo Windowed Memory")
  # demo_windowed_memory()

  _print_section("Demo Summary Memory")
  demo_summary_memory()

  # _print_section("Exercise Persistent Memory")
  # exercise_persistent_memory()

  # _print_section("Exercise Persistent Memory Proof")
  # exercise_persistent_memory_proof()
