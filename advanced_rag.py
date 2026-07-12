"""
Advanced RAG Patterns
Multi-query, self-query, compression, hybrid search
"""

import logging
import os
import uuid

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import (
  ContextualCompressionRetriever,
  EnsembleRetriever,
  ParentDocumentRetriever,
  SelfQueryRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"


def print_section(title, subtitle=None):
  """Print a section header in blue."""
  lines = "\n".join(filter(None, [title, subtitle]))
  print(f"\n\033[94m{'=' * 60}\n{lines}\n{'=' * 60}\033[0m")


# CONCISE_INSTRUCTIONS = """Make sure to answer in a concise manner,
# and if you don't know the answer, just say "I don't know."""

CONCISE_INSTRUCTIONS = """Answer ONLY using the provided context. Do not use any outside knowledge.
Be concise. If the context does not contain the answer, say "I don't know" — do not guess."""


# Sample knowledge base for demos
TECH_DOCS = [
  Document(
    page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
    metadata={
      "topic": "programming",
      "language": "python",
      "difficulty": "beginner",
    },
  ),
  Document(
    page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
    metadata={
      "topic": "programming",
      "language": "javascript",
      "difficulty": "intermediate",
    },
  ),
  Document(
    page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
    metadata={
      "topic": "ai",
      "subtopic": "machine_learning",
      "difficulty": "advanced",
    },
  ),
  Document(
    page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
    metadata={
      "topic": "ai",
      "subtopic": "llm_frameworks",
      "difficulty": "intermediate",
    },
  ),
  Document(
    page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
    metadata={
      "topic": "ai",
      "subtopic": "llm_frameworks",
      "difficulty": "advanced",
    },
  ),
  Document(
    page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
    metadata={
      "topic": "devops",
      "subtopic": "containers",
      "difficulty": "intermediate",
    },
  ),
  Document(
    page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
    metadata={
      "topic": "database",
      "type": "relational",
      "difficulty": "intermediate",
    },
  ),
  Document(
    page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
    metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
  ),
]

INFO_BURIED = [
  Document(
    page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK

Founded in 2018 by three Stanford graduates, ACME AI Solutions began as a
small consulting firm helping enterprises adopt machine learning. Our first
office was a converted garage in Palo Alto, and we had just two laptops and
a dream. The early days were challenging - we survived on instant ramen and
the occasional pizza from the client meetings.

In 2019, we secured our first major contract with a Fortune 500 retailer,
helping them build a recommendation engine. This led to rapid growth and we
moved to a proper office space in San Francisco. By 2020, we had grown to
50 employees and opened offices in Austin and Seattle.

Our current technology stack has evolved significantly over the years. For
backend services, we use Python and FastAPI. Our data pipeline runs on
Apache Spark and Airflow. For frontend, we've standardized on React and
TypeScript.

LangChain is a framework for building LLM applications. It provides tools
for prompts, chains, agents, and memory. LangChain supports multiple LLM
providers including OpenAI, Anthropic, and local models like Llama.

The company culture at ACME emphasizes work-life balance. We offer unlimited
PTO, which most employees use for an average of 25 days per year. Our
engineering teams follow agile methodology with two-week sprints.

Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We
project $70M for 2024, driven by our new enterprise AI platform. The company
went through Series B funding in 2022, raising $80M at a $500M valuation.

Employee benefits include comprehensive health insurance through Aetna, a
401(k) with 4% matching, and a generous equity package.""",
    metadata={"source": "acme_company_overview.pdf"},
  ),
  Document(
    page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4

Chapter 1: System Architecture Overview

The ACME AI Platform is built on a microservices architecture deployed on
AWS EKS (Elastic Kubernetes Service). Each microservice is containerized
using Docker and orchestrated by Kubernetes. We use Istio as our service
mesh for traffic management and observability.

Our database layer consists of PostgreSQL for transactional data, Redis
for caching, and Pinecone for vector storage. All databases are deployed
in high-availability configurations with automatic failover.

Chapter 2: Authentication and Authorization

User authentication is handled through Auth0, supporting both SSO via SAML
2.0 and OAuth 2.0 flows. We implement role-based access control (RBAC) with
four default roles: Admin, Developer, Analyst, and Viewer.

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state management, cycles and loops, human-in-the-
loop workflows, and persistence. LangGraph extends LangChain for complex
agent architectures.

Chapter 4: Monitoring and Logging

We use DataDog for application performance monitoring (APM) and log
aggregation. All services emit structured JSON logs that are collected and
indexed for searching. Alert thresholds are configured for latency (p99 >
500ms), error rates (> 1%), and resource utilization (CPU > 80%).

Chapter 5: Disaster Recovery

Our disaster recovery plan includes daily database backups stored in S3
with cross-region replication. RTO is 4 hours, and RPO is 1 hour.""",
    metadata={"source": "technical_docs_v2.4.pdf"},
  ),
]

LONG_DOC = Document(
  page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
        """,
  metadata={"source": "ai_agents_guide.md"},
)


PAGE_CONTENT = """
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
"""

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


## Common function to create a vector store for demos
def create_base_vectorstore(documents=TECH_DOCS, collection_name=None):
  """Create a vector store for demos.

  Pass documents (default TECH_DOCS) to populate it immediately, or an
  empty list to get an empty store to add to later (e.g. for
  ParentDocumentRetriever, which manages its own document additions).
  """
  print(f"\033[93mCreating vector store with {len(documents)} documents...\033[0m")
  embedding = OpenAIEmbeddings(model="text-embedding-3-small")
  collection_name = collection_name or f"demo_{uuid.uuid4().hex}"
  kwargs = {"collection_name": collection_name}

  if documents:
    return Chroma.from_documents(documents=documents, embedding=embedding, **kwargs)
  return Chroma(embedding_function=embedding, **kwargs)


## Common chain to answer a question from retrieved context, kept grounded and concise
def build_answer_chain():
  answer_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", CONCISE_INSTRUCTIONS),
      ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
  )
  return answer_prompt | LLM | StrOutputParser()


## Common helper to join retrieved docs into context and answer a question
def answer_from_docs(answer_chain, docs, question):
  context = "\n\n".join(doc.page_content for doc in docs)
  return answer_chain.invoke({"context": context, "question": question})


# Multi-query Retrieval - MultiQueryRetriever
@traceable(name="multi_query_retriever", tags=["retrieval", "multi-query"])
def demo_multi_query_retriever():
  """Multi-Query Retriever generates multiple query perspectives."""

  # Create Multi-Query Retriever
  retriever = MultiQueryRetriever.from_llm(
    retriever=create_base_vectorstore().as_retriever(search_kwargs={"k": 3}),
    llm=LLM.bind(temperature=0.3),
  )

  answer_chain = build_answer_chain()

  queries = [
    "What tools can I use to build AI applications?",
    "Show me advanced difficulty AI documents",
    # "What database topics are there?",
  ]

  for query in queries:
    print(f"\n\033[92mQuery: {query}\n\033[0m")

    # Multi-Query Retriever, Retrieving documents
    print("\033[93mMulti-Query Retriever: Retrieving Documents ...\033[0m")
    docs = retriever.invoke(query)

    # MultiQueryRetriever returns a list of non-duplicate documents.
    print(f"\033[92m\nRetrieved {len(docs)} Documents:\033[0m")
    for i, doc in enumerate(docs, start=1):
      print(f"{i}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}...")

    # Generate an answer grounded in the retrieved docs
    answer = answer_from_docs(answer_chain, docs, query)
    print(f"\n\033[93mAnswer: {answer}\033[0m")


# Self-Query Retrieval - SelfQueryRetriever
@traceable(name="self_query_retriever", tags=["retrieval", "self-query"])
def demo_self_query_retriever():
  """Self-Query Retriever lets the LLM derive metadata filters from the question."""

  metadata_field_info = [
    AttributeInfo(
      name="topic",
      description="The topic of the document. One of ['programming', 'ai', 'devops', 'database']",
      type="string",
    ),
    AttributeInfo(
      name="difficulty",
      description="The difficulty level of the document. One of ['beginner', 'intermediate', 'advanced']",
      type="string",
    ),
  ]

  document_content_description = (
    "Technical documentation covering programming languages, AI/ML frameworks, "
    "DevOps tools, and databases"
  )

  # Self-Query Retriever
  retriever = SelfQueryRetriever.from_llm(
    llm=LLM,
    vectorstore=create_base_vectorstore(),
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    search_kwargs={"k": 3},
    verbose=True,
  )

  answer_chain = build_answer_chain()

  queries = [
    "What tools can I use to build AI applications?",
    "Show me advanced difficulty AI documents",
    # "What database topics are there?",
  ]

  for query in queries:
    print(f"\n\033[92mQuery: {query}\n\033[0m")
    # Self-Query Retriever, Retrieving documents
    print("\033[93mSelf-Query Retriever: Retrieving Documents ...\033[0m")
    docs = retriever.invoke(query)

    print(f"\033[92m\nRetrieved {len(docs)} Documents:\033[0m")
    for i, doc in enumerate(docs, start=1):
      print(
        f"{i}. [{doc.metadata.get('topic')}/{doc.metadata.get('difficulty')}] {doc.page_content[:80]}..."
      )

    # Generate an answer grounded in the retrieved docs
    answer = answer_from_docs(answer_chain, docs, query)
    print(f"\n\033[93mAnswer: {answer}\033[0m")


# Contextual Compression Retriever - ContextualCompressionRetriever
@traceable(name="contextual_compression_retriever", tags=["retrieval", "compression"])
def demo_contextual_compression():
  """Contextual Compression extracts only relevant parts."""

  query = "What frameworks exist for building LLM applications?"
  print(f"\n\033[92mQuery: {query}\n\033[0m")

  # Create a vector store with documents that have a lot of buried information
  vectorstore = create_base_vectorstore(documents=INFO_BURIED, collection_name="compression_demo")
  answer_chain = build_answer_chain()

  # Without compression
  print("\n\033[95m--- WITHOUT Compression (full chunks) ---\033[0m")
  base_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
  print(f"\033[92m\nRetrieved {len(base_docs)} Documents:\033[0m")
  for doc in base_docs:
    print(f"Length: {len(doc.page_content)} chars")
    print(f"Content: {doc.page_content[:150]}...\n")

  base_answer = answer_from_docs(answer_chain, base_docs, query)
  print(f"\n\033[93mAnswer (without compression): {base_answer}\033[0m")

  # With compression
  # Wrap retriever with compression
  print("\n\033[95m--- WITH Compression (relevant only) ---\033[0m")
  compression_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(LLM),
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
  )
  compressed_docs = compression_retriever.invoke(query)
  print(f"\033[92m\nRetrieved {len(compressed_docs)} Documents:\033[0m")
  for doc in compressed_docs:
    print(f"Length: {len(doc.page_content)} chars")
    print(f"Content: {doc.page_content}\n")

  compressed_answer = answer_from_docs(answer_chain, compressed_docs, query)
  print(f"\n\033[93mAnswer (with compression): {compressed_answer}\033[0m")


# Hybrid Search - EnsembleRetriever
@traceable(name="ensemble_hybrid_search", tags=["retrieval", "hybrid", "bm25"])
def demo_ensemble_hybrid_search():
  """Hybrid search combining keyword (BM25) and semantic search."""

  # BM25 keyword retriever
  bm25_retriever = BM25Retriever.from_documents(TECH_DOCS, k=3)
  # Semantic retriever
  semantic_retriever = create_base_vectorstore().as_retriever(search_kwargs={"k": 3})

  # Ensemble combines both
  ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
  )

  answer_chain = build_answer_chain()

  # Test queries
  # queries = [
  #     "PostgreSQL pgvector",  # Keyword-heavy (BM25 helps)
  #     "What database stores embeddings?",  # Semantic (vectors help)
  # ]
  queries = [
    "PostgreSQL pgvector",  # Keyword-heavy (BM25 helps)
    "What database stores embeddings?",  # Semantic (vectors help)
    "ACID transactions",  # Keyword-heavy (BM25 helps)
    "How do I store AI model outputs for later retrieval?",  # Semantic (vectors help)
    "fast similarity lookup for embeddings",  # Mixed
    "What is BM25 in Chinese?",  # Keyword-heavy (BM25 helps)
    "What is Claude?",  # Semantic (vectors help)
  ]

  for query in queries:
    print(f"\n\033[92mQuery: {query}\n\033[0m")
    # print("-" * 40)

    # Compare results
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    ensemble_results = ensemble_retriever.invoke(query)

    print(f"\n1. BM25 {len(bm25_results)} docs. Top result: {bm25_results[0].page_content[:60]}...")
    print(
      f"2. Semantic {len(semantic_results)} docs. Top result: {semantic_results[0].page_content[:60]}..."
    )
    print(
      f"3. Ensemble {len(ensemble_results)} docs. Top result: {ensemble_results[0].page_content[:60]}...\n"
    )

    # Generate an answer grounded in the ensemble results
    answer = answer_from_docs(answer_chain, ensemble_results, query)
    print(f"\n\033[93mAnswer: {answer}\033[0m")


# Parent Document Retriever - ParentDocumentRetriever
@traceable(name="parent_document_retriever", tags=["retrieval", "parent-document"])
def demo_parent_document_retriever():
  """Parent Document Retriever: small chunks for search, large for context."""

  # Splitters
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

  # Storage
  # ParentDocumentRetriever 要求 vectorstore 一開始是空的
  vectorstore = create_base_vectorstore(
    documents=[],
    collection_name="parent_child_demo",
  )
  store = InMemoryStore()

  # Create retriever
  retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3},
  )

  answer_chain = build_answer_chain()

  # Search
  questions = [
    "What is LangGraph used for?",
    "What are the key components of LangChain?",
    "What are the differences between LangChain and LangGraph?",
    # "What are the production considerations for deploying agents?",
    "What is Claude?",
  ]

  # Info Buried and Long documents to demonstrate parent/child splitting
  retriever.add_documents(INFO_BURIED)
  retriever.add_documents([LONG_DOC])

  for query in questions:
    print(f"\n\033[92mQuery: {query}\n\033[0m")

    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=3)
    print(f"\n\033[95m--- Child Chunks: {len(child_docs)} (what search found) ---\033[0m")

    # Check child docs and their parent IDs
    for index, doc in enumerate(child_docs, start=1):
      doc_id = doc.metadata.get("doc_id")

      print(
        f"\n{index}. Doc ID: {doc_id}, Len: {len(doc.page_content)} chars"
        f"\n   Content: {doc.page_content[:80]}..."
      )

    unique_doc_ids = {
      doc.metadata.get("doc_id") for doc in child_docs if doc.metadata.get("doc_id") is not None
    }

    print(f"\033[38;5;208m\nChild Unique Doc IDs Len: {len(unique_doc_ids)}\033[0m")
    print(f"\033[38;5;208mChild Unique Doc IDs: {unique_doc_ids}\n\033[0m")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n\033[95m--- Parent Chunks: {len(parent_docs)} (what's returned) ---\033[0m")
    # print(
    #   f"Parent Content Preview: {len(parent_docs[0].page_content)} chars, {parent_docs[0].page_content[:300]}..."
    # )

    for index, doc in enumerate(parent_docs, start=1):
      print(
        f"\n{index}. Len: {len(doc.page_content)} chars"
        f"\n   Content Preview: {doc.page_content[:80]}..."
      )

    # Generate an answer via the Pa
    answer = answer_from_docs(answer_chain, parent_docs, query)
    print(f"\n\033[93mAnswer: {answer}\033[0m")


# Complete RAG chain with advanced retrieval - Multi-query + Compression + RAG
@traceable(name="advanced_rag_chain", tags=["rag", "multi-query", "compression"])
def demo_advanced_rag_chain():
  """Complete RAG chain with advanced retrieval."""

  # Multi-query for better recall => Compression to focus on relevant info
  advanced_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(LLM),
    base_retriever=MultiQueryRetriever.from_llm(
      retriever=create_base_vectorstore().as_retriever(search_kwargs={"k": 3}), llm=LLM
    ),
  )

  # RAG prompt
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", CONCISE_INSTRUCTIONS),
      (
        "human",
        """Answer the question based on the following context. Be specific and cite which technologies you're referring to.

Context:
{context}

Question: {question}

Answer:""",
      ),
    ]
  )

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  # Build chain
  rag_chain = (
    {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
  )

  # Test
  questions = [
    "What options do I have for building AI agents?",
    "How can I store and search embeddings?",
    # "What is Claude and how does it compare to ChatGPT?",
    "What is BOLLETTT?",
  ]

  for q in questions:
    print(f"\n\033[92mQuery: {q}\n\033[0m")
    answer = rag_chain.invoke(q)
    print(f"\n\033[93mAnswer: {answer}\033[0m")


if __name__ == "__main__":
  print_section("MULTI-QUERY RETRIEVAL")
  demo_multi_query_retriever()

  print_section("SELF-QUERY RETRIEVAL")
  demo_self_query_retriever()

  print_section("CONTEXTUAL COMPRESSION RETRIEVER")
  demo_contextual_compression()

  print_section("ENSEMBLE/HYBRID RETRIEVER")
  demo_ensemble_hybrid_search()

  print_section("PARENT DOCUMENT RETRIEVER")
  demo_parent_document_retriever()

  print_section("ADVANCED RAG DEMO: Multi-query + Compression + RAG")
  demo_advanced_rag_chain()
