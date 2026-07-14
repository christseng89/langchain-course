"""
Section 2 Project: AI Research Assistant
Complete RAG system with conversation memory
"""

from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.chat_history import (
  BaseChatMessageHistory,
  InMemoryChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================
# Data Models
# ============================================================
class StructuredResponse(BaseModel):
  """Structured response from the research assistant."""

  answer: str = Field(description="The answer to the question")
  confidence: str = Field(description="high, medium, or low based on source quality")
  sources: List[str] = Field(description="List of source documents used")
  key_quotes: List[str] = Field(description="Relevant quotes from sources", default=[])
  follow_up_questions: List[str] = Field(description="Suggested follow-up questions")


# ============================================================
# Prompts
# ============================================================
CONCISE_INSTRUCTIONS = """Answer ONLY using the provided context. Do not use any outside knowledge.
Be concise. If the context does not contain the answer, say "I don't know" — do not guess."""


PROMPT = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      f"""You are an AI Research Assistant. Analyze the provided documents
and return a structured response. {CONCISE_INSTRUCTIONS}

Rules:
1. ONLY use information from the provided context
2. If the context doesn't have the answer, say so in the answer field
3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
4. Include the source filenames you actually used
5. Extract key quotes word-for-word from the context
6. Suggest 2-3 follow-up questions the user might want to ask

Use conversation history to understand follow-up questions.""",
    ),
    MessagesPlaceholder(variable_name="history"),
    (
      "human",
      """Context documents:

{context}

Available sources: {sources}

Question: {question}""",
    ),
  ]
)


PROMPT2 = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      f"""You are an AI Research Assistant. Answer questions
based ONLY on the provided context documents. {CONCISE_INSTRUCTIONS}

Rules:
1. Only use information from the context below
2. If the context doesn't have the answer, say so
3. Cite which sources you used (e.g. "According to Source 1...")
4. Rate your confidence: high, medium, or low""",
    ),
    MessagesPlaceholder(variable_name="history"),
    (
      "human",
      """Context documents:

{context}

Question: {question}

Provide a clear answer with source citations.""",
    ),
  ]
)


TEXT_NATURAL_NETWORKS = """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """

TEXT_RAG = """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """

TEXT_LANGCHAIN = """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """


def print_section(name: str) -> None:
  blue = "\033[94m"
  reset = "\033[0m"
  print(f"\n{blue}{'#' * 60}\n# {name}\n{'#' * 60}{reset}\n")


# ============================================================
# Research Assistant Class
# ============================================================


class AIResearchAssistant:
  """AI Research Assistant with document ingestion and retrieval."""

  def __init__(
    self,
    persist_directory: str = "./research_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
  ):
    self.persist_directory = persist_directory
    # 1. Embeddings - turns text into vectors
    self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print(f"\033[93mEmbedding model: {self.embeddings.model}\033[0m")
    print(f"\033[93mLLM model: {self.llm.model_name}\033[0m")
    # 2. Splitter - breaks big docs into chunks
    self.splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["\n\n", "\n", ". ", " ", ""],
    )
    # 3. Vector store - stores and searches embeddings
    self.vectorstore = Chroma(
      persist_directory=persist_directory,
      embedding_function=self.embeddings,
      collection_name="research_docs",
    )

    self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

    print_section("STEP 0: Research Assistant initialized")
    print(f"Vector store: {persist_directory}")
    print(f"Documents indexed: {self.vectorstore._collection.count()}\n")

  def add_documents(
    self,
    documents: List[Document],
    source_name: Optional[str] = None,
  ) -> int:
    """Add documents to the research database."""

    # Tag with source name
    if source_name:
      for doc in documents:
        doc.metadata["source"] = source_name

    # Split into chunks
    chunks = self.splitter.split_documents(documents)

    # Timestamp each chunk
    for chunk in chunks:
      chunk.metadata["indexed_at"] = datetime.now().isoformat()

    # Store in vector DB
    self.vectorstore.add_documents(chunks)

    print(f"Added {len(chunks)} chunks from {len(documents)} documents")
    return len(chunks)

  def add_text(self, text: str, source: str, metadata: dict = None) -> int:
    """Add a single text string as a document."""
    doc = Document(page_content=text, metadata={"source": source, **(metadata or {})})
    return self.add_documents([doc])

  def get_document_count(self) -> int:
    """Get total number of indexed chunks."""
    return self.vectorstore._collection.count()

  def list_sources(self) -> List[str]:
    """List all unique sources in the database."""
    results = self.vectorstore._collection.get()
    sources = set()
    for metadata in results.get("metadatas", []):
      if metadata and "source" in metadata:
        sources.add(metadata["source"])
    return sorted(list(sources))

  def _build_retriever(self, use_advanced: bool = False):
    """Build retriever -- basic or advanced (multi-query + compression)"""

    # Base: simple similarity search
    base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    if not use_advanced:
      return base_retriever

    # Compression: LLM strips each retrieved chunk down to the relevant part
    compressor = LLMChainExtractor.from_llm(self.llm)
    compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor,
      # Multi-query: LLM generates multiple search queries
      base_retriever=MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=self.llm,
      ),
    )

    return compression_retriever

  def _format_docs_for_context(self, docs) -> str:
    """Format retrieved documents into a string for the prompt."""
    if not docs:
      return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs):
      source = doc.metadata.get("source", "Unknown")
      formatted.append(f"[Source {i + 1}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

  def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
    """Get or create session history."""
    if session_id not in self.session_store:
      self.session_store[session_id] = InMemoryChatMessageHistory()
    return self.session_store[session_id]

  def ask_structured(
    self,
    question: str,
    session_id: str = "default",
    use_advanced: bool = True,
  ) -> StructuredResponse:
    """Ask a question and get a structured response."""

    # LLM that returns a Pydantic object instead of a string
    structured_llm = self.llm.with_structured_output(StructuredResponse)

    # Get memory
    history = self._get_session_history(session_id)

    # Retrieve
    retriever = self._build_retriever(use_advanced=use_advanced)
    docs = retriever.invoke(question)
    context = self._format_docs_for_context(docs)
    sources = list(set(d.metadata.get("source", "Unknown") for d in docs))

    chain = PROMPT | structured_llm

    response = chain.invoke(
      {
        "context": context,
        "question": question,
        "sources": ", ".join(sources),
        "history": (history.messages[-10:] if hasattr(history, "messages") else history[-10:]),
      }
    )

    # Save to memory (store just the answer text)
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=response.answer))

    return response

  def ask(
    self,
    question: str,
    session_id: str = "default",
    use_advanced: bool = True,
  ) -> str:
    """Ask a question against the research documents."""

    history = self._get_session_history(session_id)

    # Use basic or advanced retriever
    retriever = self._build_retriever(use_advanced=use_advanced)
    docs = retriever.invoke(question)
    context = self._format_docs_for_context(docs)

    # Step 4: Build and run the chain
    prompt = PROMPT2
    chain = prompt | self.llm | StrOutputParser()

    response = chain.invoke(
      {
        "context": context,
        "question": question,
        "history": history.messages[-10:],  # Last 10 messages for context
      }
    )

    # save this Q&A to history
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=response))

    return response


def print_research_response(question: str, response: StructuredResponse):
  """Pretty print a structured research response."""

  print(f"\n\033[92mQuery: {question}\033[0m")
  print(f"\033[93mAnswer: {response.answer}\033[0m")
  print(
    f"\033[38;5;208mConfidence: {response.confidence}, Sources: {', '.join(response.sources)}\033[0m\n"
  )

  if response.key_quotes:
    print("\033[95mKey Quotes:\033[0m")
    for i, q in enumerate(response.key_quotes):
      print(f"{i + 1}. {q[:120]}")

  if response.follow_up_questions:
    print("\n\033[93mFollow-up Questions:\033[0m")
    for i, fq in enumerate(response.follow_up_questions):
      print(f"{i + 1}. {fq}")

  # print("\n")


if __name__ == "__main__":
  import shutil

  shutil.rmtree("./research_db", ignore_errors=True)
  assistant = AIResearchAssistant()

  # Add research docs
  assistant.add_text(TEXT_NATURAL_NETWORKS, source="attention_mechanisms.pdf")
  assistant.add_text(TEXT_RAG, source="rag_survey.pdf")
  assistant.add_text(TEXT_LANGCHAIN, source="langchain_docs.md")

  print(f"\nIndexed: {assistant.get_document_count()} chunks")

  session = "structured_demo"

  # --- Step 1: String vs Structured comparison ---
  print_section("STEP 1: String response vs Structured response")

  question = "What is RAG and what are its benefits?"
  print(f"\033[92mQuery: {question}\033[0m")

  print("\n\033[38;5;208m--- String response ---\033[0m")
  string_response = assistant.ask(question, "string_test")
  print(f"Type: {type(string_response)}")
  print(f"Response: {string_response[:200]}...")

  print("\n\033[38;5;208m--- Structured response ---\033[0m")
  structured_response = assistant.ask_structured(question, "struct_test")
  print(f"Type: {type(structured_response)}")
  print(f"Answer: {structured_response.answer[:100]}...")
  print(f"Confidence: {structured_response.confidence}")
  print(f"Sources: {structured_response.sources}")
  print(f"Key_quotes: {structured_response.key_quotes[:2]}")
  print(f"Follow_up_questions: {structured_response.follow_up_questions}")

  # --- Step 2: Access fields directly ---
  print_section("STEP 2: Use fields in your code - Structured response")

  question = "What is the attention mechanism?"
  print(f"\033[92mQuery: {question}\033[0m")

  response = assistant.ask_structured(question, session)
  # This is what your app code looks like
  print(f"\033[93mAnswer: {response.answer}\033[0m")
  if response.confidence == "high":
    print(f"\033[38;5;208mConfident Answer from: {', '.join(response.sources)}\033[0m")
  else:
    print("\033[38;5;208mLow Confidence Answer -- may need more sources\033[0m")

  print("\n\033[38;5;208mSuggested follow-ups:\033[0m")
  for i, fq in enumerate(response.follow_up_questions):
    print(f"{i + 1}. {fq}")

  # --- Step 3: Multi-turn with structured output ---
  print_section("STEP 3: Memory works with structured output too")

  questions = [
    "What are the components of RAG?",
    "How does the second component work?",
    "Connect everything we discussed to LangChain.",
    "What is my name?",
    "What is Claude?",
  ]

  for i, q in enumerate(questions):
    r = assistant.ask_structured(q, session)
    print_research_response(q, r)

  # --- Step 4: Final stats ---
  print_section("STEP FINAL: What we built across 5 videos")
  print("\033[92mAssistant AI Final Results:\033[0m")

  history = assistant._get_session_history(session)
  msg_count = len(history.messages) if hasattr(history, "messages") else len(history)

  print(
    f"""
Document ingestion    -> {assistant.get_document_count()} chunks indexed
Sources tracked       -> {assistant.list_sources()}
Basic retrieval       -> similarity search
Advanced retrieval    -> Multi-query + Compression
Conversation memory   -> {msg_count} messages in session '{session}'
Structured output     -> StructuredResponse with {len(StructuredResponse.model_fields)} fields

From raw text to a production-ready research assistant.
That's the full RAG pipeline.
    """
  )

  # Cleanup
  shutil.rmtree("./research_db", ignore_errors=True)
