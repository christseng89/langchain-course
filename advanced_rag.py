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


CONCISE_INSTRUCTIONS = """Make sure to answer in a concise manner,
and if you don't know the answer, just say "I don't know."""

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

  vectorstore = create_base_vectorstore()
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
    "What is Claude?",  # Semantic (vectors help)
  ]

  for query in queries:
    print(f"\n\033[92mQuery: {query}\n\033[0m")
    # print("-" * 40)

    # Compare results
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)
    ensemble_results = ensemble_retriever.invoke(query)

    print(f"\nBM25 {len(bm25_results)} docs. Top result: {bm25_results[0].page_content[:60]}...")
    print(
      f"Semantic {len(semantic_results)} docs. Top result: {semantic_results[0].page_content[:60]}..."
    )
    print(
      f"Ensemble {len(ensemble_results)} docs. Top result: {ensemble_results[0].page_content[:60]}..."
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
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

  # Storage
  vectorstore = create_base_vectorstore(documents=[], collection_name="parent_child_demo")
  store = InMemoryStore()

  # Create retriever
  retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
  )

  answer_chain = build_answer_chain()

  # Long document to demonstrate parent/child splitting
  long_doc = Document(
    page_content=PAGE_CONTENT,
    metadata={"source": "ai_agents_guide.md"},
  )
  # Add document
  retriever.add_documents([long_doc])

  # Search
  questions = [
    "What is LangGraph used for?",
    "What are the key components of LangChain?",
    # "What are the differences between LangChain and LangGraph?",
    "What are the production considerations for deploying agents?",
    "What is Claude?",
  ]

  for query in questions:
    print(f"\n\033[92mQuery: {query}\n\033[0m")

    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=1)
    print(f"\n\033[95m--- Child Chunk {len(child_docs)} docs (what search found) ---\033[0m")
    print(
      f"Child Content [0]: {len(child_docs[0].page_content)} chars, {child_docs[0].page_content}...\n"
    )

    # Generate an answer grounded in the child chunk
    answer = answer_from_docs(answer_chain, child_docs, query)
    print(f"\n\033[93mChild Answer: {answer}\n\033[0m")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n\033[95m--- Parent Chunk {len(parent_docs)} docs (what's returned) ---\033[0m")
    print(
      f"Parent Content Preview: {len(parent_docs[0].page_content)} chars, {parent_docs[0].page_content[:300]}..."
    )

    # Generate an answer grounded in the parent chunk
    answer = answer_from_docs(answer_chain, parent_docs, query)
    print(f"\n\033[93mParent Answer: {answer}\033[0m")


# Complete RAG chain with advanced retrieval - Multi-query + Compression + RAG
@traceable(name="advanced_rag_chain", tags=["rag", "multi-query", "compression"])
def demo_advanced_rag_chain():
  """Complete RAG chain with advanced retrieval."""

  # Multi-query for better recall
  multi_retriever = MultiQueryRetriever.from_llm(
    retriever=create_base_vectorstore().as_retriever(search_kwargs={"k": 3}), llm=LLM
  )

  # Compression to focus on relevant info
  advanced_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(LLM), base_retriever=multi_retriever
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
    # "How can I store and search embeddings?",
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
