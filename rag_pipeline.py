"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

import tempfile
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
print(f"\033[93m\nEmbedding model: {embeddings_model.model}\033[0m")

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

LLM = init_chat_model(model="gpt-4o-mini", temperature=0.2)
print(f"\033[93mLLM model: {LLM.model_name}\033[0m")


def print_section(title):
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


CONCISE_INSTRUCTIONS = """Make sure to answer in a concise manner,
and if you don't know the answer, just say "I don't know."""


def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_sources(docs):
  formatted = []
  for i, doc in enumerate(docs):
    source = doc.metadata.get("source", "unknown")
    formatted.append(f"[{i + 1}] {source}:\n{doc.page_content}")
  return "\n\n".join(formatted)


def create_kb():
  """Create a vector store from knowledge base."""

  # split the knowledge base into chunks
  print("\033[93mCreating Vector Store from Knowledge Base\033[0m")
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"})

  chunks = splitter.split_documents([doc])

  # create a vector store from the chunks
  vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=tempfile.mkdtemp(),
  )
  return vector_store


# Create a global vector store for the knowledge base
VECTOR_STORE = create_kb()


# Basic RAG Demo
def demo_basic_rag():

  # Retriever
  retriever = VECTOR_STORE.as_retriever(search_type="similarity", search_kwargs={"k": 2})
  # RAG Prompt Template
  prompt = ChatPromptTemplate.from_template(
    f"""
Answer the question based only on the following context:

{{context}}

Question: {{question}}

Answer:


{CONCISE_INSTRUCTIONS}"""
  )

  # Rag chain
  rag_chain = (
    {
      "context": retriever | format_docs,
      # Context is 先 retriever，再 format_docs 的結果。
      # context = format_docs(retriever(question))
      "question": RunnablePassthrough(),  # Question is passed through as-is
    }
    | prompt
    | LLM
    | StrOutputParser()
  )

  # Test the RAG chain
  # Test
  questions = [
    "What is LangChain?",
    "Who created LangChain?",
    "What is LangGraph used for?",
    "What is LangSmith?",
    "What is the pricing for LangSmith?",
    "What is Claude?",
  ]

  for q in questions:
    answer = rag_chain.invoke(q)
    print(f"\033[92mQ: {q}\033[0m")
    print(f"A: {answer}\n")


# RAG with Sources Demo
def demo_rag_with_sources():

  # vectorstore = create_kb()
  retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
  prompt = ChatPromptTemplate.from_template(
    f"""
Answer the question based on the context below. Include which sources you used.

Context:
{{context}}

Question: {{question}}

Answer (include sources):


{CONCISE_INSTRUCTIONS}"""
  )

  rag_chain = (
    {
      "context": retriever | format_docs_with_sources,
      "question": RunnablePassthrough(),
    }
    | prompt
    | LLM
    | StrOutputParser()
  )

  questions = [
    "What are the core components of LangChain?",
    "What is LangGraph used for?",
    "What is the pricing for LangSmith?",
    "What is Claude Code?",
  ]

  for q in questions:
    answer = rag_chain.invoke(q)
    print(f"\033[92mQ: {q}\033[0m")
    print(f"A: {answer}\n")


# RAG with Fallback Demo
def demo_rag_with_fallback():
  retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 2})

  prompt = ChatPromptTemplate.from_template(
    f"""
Answer the question based ONLY on the following context.

{CONCISE_INSTRUCTIONS}

Context:
{{context}}

Question: {{question}}

Answer:
"""
  )

  # Same prompt without CONCISE_INSTRUCTIONS, to compare fallback behavior
  prompt1 = ChatPromptTemplate.from_template(
    """
Answer the question based ONLY on the following context.

Context:
{context}

Question: {question}

Answer:
"""
  )

  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
  )

  rag_chain1 = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt1
    | LLM
    | StrOutputParser()
  )

  questions = [
    "What is the pricing for LangSmith?",  # In knowledge base
    "What is the stock price of OpenAI?",  # Not in knowledge base
    "How do I deploy LangChain to AWS?",  # Not in knowledge base
  ]

  for q in questions:
    answer = rag_chain.invoke(q)
    answer1 = rag_chain1.invoke(q)
    print(f"\033[92mQ: {q}\033[0m")
    print(f"A (with concise instructions): {answer}")
    print(f"A (without concise instructions): {answer1}\n")


# RAG with Structured Output Demo
def demo_structured_rag():
  """RAG with structured output."""

  retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})

  class RAGResponse(BaseModel):
    """Structured RAG response."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low")
    sources_used: List[str] = Field(description="List of sources referenced")
    follow_up: List[str] = Field(description="Suggested follow-up question")

  # Use the structured output LLM
  structured_llm = LLM.with_structured_output(RAGResponse)

  prompt = ChatPromptTemplate.from_template(
    f"""
Based on the context below, answer the question.

Context:
{{context}}

Question: {{question}}

Provide a structured response.


{CONCISE_INSTRUCTIONS}"""
  )

  rag_chain = (
    {
      "context": retriever
      | format_docs_with_sources,  # Context is retrieved and formatted w sources
      "question": RunnablePassthrough(),  # Question
    }
    | prompt
    | structured_llm
  )

  question = [
    "What is LangGraph?",
    "What are the core components of LangChain?",
    "What is Claude Code?",
  ]

  result = rag_chain.invoke("What is LangGraph?")

  for q in question:
    result = rag_chain.invoke(q)
    print(f"\033[92mQ: {q}\033[0m")
    print(f"A: {result.answer}\n")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}\n")


# Exercise: Build a document Q&A system
def exercise_document_qa():
  """
  EXERCISE: Build a complete document Q&A system that:
  1. Takes a text document as input
  2. Splits and embeds it
  3. Allows multiple questions
  4. Returns answers with confidence scores
  """

  class DocumentQA:
    def __init__(self, document: str, source_name: str = "document"):
      # Split document
      splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      doc = Document(page_content=document, metadata={"source": source_name})
      chunks = splitter.split_documents([doc])

      # Create vector store
      self.vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
      )
      self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

      self.prompt = ChatPromptTemplate.from_template(
        f"""
Answer based on the context. Rate your confidence (high/medium/low).

Context: {{context}}
Question: {{question}}

Format: [Confidence: X] Answer


{CONCISE_INSTRUCTIONS}"""
      )

      self.chain = (
        {
          "context": self.retriever | format_docs,
          "question": RunnablePassthrough(),
        }
        | self.prompt
        | LLM
        | StrOutputParser()
      )

    def ask(self, question: str) -> str:
      return self.chain.invoke(question)

  # Test
  test_doc = """
    The Python programming language was created by Guido van Rossum.
    First released in 1991, Python emphasizes code readability.
    Python 3.12 was released in October 2023 with improved error messages.
    The language is named after Monty Python, not the snake.
    """

  qa = DocumentQA(test_doc, "python_facts")  # "python_facts" = source name

  questions = [
    "What is Python?",
    "Who created Python?",
    "When was Python 3.12 released?",
    "Why is Python named Python?",
    "What is Claude Code?",  # Not in document
  ]

  for q in questions:
    answer = qa.ask(q)
    print(f"\033[92mQ: {q}\033[0m")
    print(f"A: {answer}\n")


if __name__ == "__main__":
  print_section("Basic RAG Demo")
  demo_basic_rag()

  print_section("RAG with Sources Demo")
  demo_rag_with_sources()

  print_section("RAG with Fallback Demo")
  demo_rag_with_fallback()

  print_section("Structured RAGResponse Demo")
  demo_structured_rag()

  print_section("Exercise: Document Q&A System")
  exercise_document_qa()
