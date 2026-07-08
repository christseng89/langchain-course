import gc
import tempfile
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
print(f"\033[93m\nEmbedding model: {embeddings_model.model}\033[0m")

# Sample documents
SAMPLE_DOCS = [
  Document(
    page_content="LangChain is a framework for developing applications powered by language models.",
    metadata={"source": "langchain_docs", "topic": "overview"},
  ),
  Document(
    page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    metadata={"source": "langgraph_docs", "topic": "overview"},
  ),
  Document(
    page_content="Vector stores are databases optimized for storing and searching embeddings.",
    metadata={"source": "vector_guide", "topic": "database"},
  ),
  Document(
    page_content="RAG combines retrieval with generation for more accurate LLM responses.",
    metadata={"source": "rag_guide", "topic": "architecture"},
  ),
  Document(
    page_content="Embeddings convert text into numerical vectors for semantic similarity.",
    metadata={"source": "embeddings_guide", "topic": "fundamentals"},
  ),
  Document(
    page_content="Chroma is an open-source embedding database for AI applications.",
    metadata={"source": "chroma_docs", "topic": "database"},
  ),
  Document(
    page_content="FAISS is a library for efficient similarity search developed by Facebook.",
    metadata={"source": "faiss_docs", "topic": "database"},
  ),
  Document(
    page_content="Pinecone is a managed vector database service for production workloads.",
    metadata={"source": "pinecone_docs", "topic": "database"},
  ),
]

PERSIST_DIR = "./chroma_db/"


def _close_chroma(vectorstore):
  """Stop Chroma's underlying system to release its sqlite file handles.

  On Windows, chromadb caches the System object internally even after
  `vectorstore` goes out of scope, so the temp dir's files stay locked
  unless we stop it explicitly before cleanup runs.
  """
  vectorstore._client._system.stop()
  gc.collect()


def print_section(title):
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


@contextmanager
def temp_chroma_vectorstore(documents=SAMPLE_DOCS):
  """Create a Chroma vector store in a temp dir, yield it, then clean it up."""
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
    # Using SAMPLE_DOCS as default, but can be overridden for testing
    vectorstore = Chroma.from_documents(
      documents=documents, embedding=embeddings_model, persist_directory=tmpdir
    )
    try:
      yield vectorstore
    finally:
      _close_chroma(vectorstore)


def chroma_basics():
  with temp_chroma_vectorstore() as vectorstore:
    print(f"Vector store created {vectorstore._collection.count()} documents and persisted.\n")

    # perform similarity search
    query = "What is LangChain?"
    print(f"\033[92mQuery: `{query}`\n\033[0m")

    top_k = 2
    results = vectorstore.similarity_search(query, k=top_k)

    print(f"Top `{top_k}` results:")
    for i, doc in enumerate(results):
      print(f" - {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")


def similarity_search_with_scores():
  with temp_chroma_vectorstore() as vectorstore:
    # perform similarity search with scores
    query = "Explain vector stores."
    print(f"\033[92mQuery: `{query}`\n\033[0m")

    top_k = 3
    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    print(f"Top `{top_k}` results with scores:\n")
    for i, (doc, score) in enumerate(results_with_scores):
      final_score = 1 / (1 + score)  # Convert distance to similarity
      print(
        f" - {i + 1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
      )


def metadata_filtering():
  with temp_chroma_vectorstore() as vectorstore:
    top_k = 5
    query = "What databases are available?"
    print(f"\033[92mQuery: `{query}`\n\033[0m")

    # without metadata filtering
    results = vectorstore.similarity_search(query, k=top_k)
    print(f"Top `{top_k}` results without metadata filtering:\n")
    for i, doc in enumerate(results):
      print(f" - {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")

    # with metadata filtering
    filter_criteria = {"topic": "database"}
    filtered_results = vectorstore.similarity_search(query, k=top_k, filter=filter_criteria)
    print(f"\nTop `{top_k}` results with metadata filtering `{filter_criteria['topic']}`:")
    for i, doc in enumerate(filtered_results):
      print(f" - {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")

    print(
      f"\033[93m\nResults == Filtered Results: {results == filtered_results}\033[0m"
    )  # Should be False, as filtering changes results


# Vector Store as Retriever
def vs_as_retriever(search_kwargs={"k": 3}, fetch_k=5):
  with temp_chroma_vectorstore() as vectorstore:
    # Similarity Vector Store retriever usage
    # top_k = 3
    query = "How do I build AI applications?"
    print(f"\033[92mQuery: `{query}`\n\033[0m")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    # use retriever to get relevant documents
    docs = retriever.invoke(query)

    print(f"Top k = `{search_kwargs['k']}` Similarity Vector Store retriever results:")
    for i, doc in enumerate(docs):
      print(f" - {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")

    # MMR Vector Store retriever usage MMR 会在保证相关的前提下，尽量避免返回内容重复的结果

    # MMR (Maximal Marginal Relevance) retriever usage
    # Add fetch_k to search_kwargs for MMR, which fetches more candidates and returns k diverse results
    mmr_search_kwargs = {
      **search_kwargs,
      "fetch_k": fetch_k,
    }  # fetch_k candidates, return k diverse
    mmr_retriever = vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs=mmr_search_kwargs,  # fetch_k candidates, return k diverse
    )
    mmr_docs = mmr_retriever.invoke(query)
    print(
      f"\nTop k = `{mmr_search_kwargs['k']}` and fetch_k = `{mmr_search_kwargs['fetch_k']}` MMR Vector Store retriever results:"
    )
    for i, doc in enumerate(mmr_docs):
      print(f" - {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")


# Exercise: Persist Chroma Vector Store
def persist_chroma():
  # Using SAMPLE_DOCS
  vectorstore = Chroma.from_documents(
    documents=SAMPLE_DOCS,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
  )

  # verify search still works
  top_k = 2
  query = "LangChain"
  print(f"\033[92mQuery: `{query}`\n\033[0m")

  original_count = vectorstore._collection.count()
  print(f"Persisted vector store with {original_count} documents.")
  print(f"Vector store persisted at: {PERSIST_DIR}")

  results = vectorstore.similarity_search(query, k=top_k)
  print(f"Top `{top_k}` vector search result: {results[0].page_content[:50]}...\n")

  # simulate restart - load from disk, 证明 Chroma 的持久化（persistence）确实有效
  del vectorstore

  reloaded = Chroma(
    embedding_function=embeddings_model,
    persist_directory=PERSIST_DIR,
  )

  reloaded_count = reloaded._collection.count()
  print(f"Reloaded vector store with {reloaded_count} documents.")
  print(f"Reloaded vector store from: {PERSIST_DIR}")
  # verify search still works
  # top_k = 2
  results = reloaded.similarity_search(query, k=top_k)
  print(f"Top `{top_k}` reload search result: {results[0].page_content[:50]}...")


# Exercise: Set up Chroma + retriever
def exercise_vector_store_setup(search_type="similarity", fetch_k=10):
  """
  EXERCISE: Create a complete vector store setup that:
  1. Takes a list of text strings
  2. Splits them into chunks
  3. Stores in Chroma
  4. Returns a configured retriever

  Test with sample documents.
  """

  def create_retriever(
    texts: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    top_k: int = 3,
    search_type: str = "similarity",
    fetch_k: int = 10,
  ):

    # Create documents
    docs = [Document(page_content=t) for t in texts]

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    # Create vector store (in-memory for exercise) using split docs
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings_model)

    # Return retriever
    search_kwargs = {"k": top_k}
    if search_type == "mmr":
      search_kwargs["fetch_k"] = fetch_k

    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

  # Test the function
  # Test
  sample_texts = [
    "Python is a versatile programming language used in web development, "
    "data science, machine learning, and automation. It has a simple syntax "
    "that makes it easy to learn and read.",
    "JavaScript is the language of the web. It runs in browsers and on "
    "servers with Node.js. Modern frameworks like React and Vue make "
    "building web applications efficient.",
    "Rust is a systems programming language focused on safety and "
    "performance. It prevents common bugs like null pointer dereferences "
    "and data races at compile time.",
  ]

  retriever = create_retriever(
    texts=sample_texts,
    chunk_size=200,
    chunk_overlap=20,
    top_k=2,
    search_type=search_type,
    fetch_k=fetch_k,
  )

  # Test the VectorStore as a Retriever with queries
  print(f"Testing retriever with search_type='{search_type}':")
  queries = [
    "What's good for web development?",
    "Which language is safest?",
  ]
  for query in queries:
    print(f"\033[92m\nQuery: `{query}`\n\033[0m")
    results = retriever.invoke(query)
    for doc in results:
      print(f" - {doc.page_content[:120]}...")


if __name__ == "__main__":
  # Vector Store 是具体实现、功能全（含 score、filter 等），但接口因后端而异。
  # Retriever 是标准化的 Runnable 包装，牺牲一些细粒度控制换来可插拔性，方便接入 LangChain 的 chain/graph 体系。

  print_section("Chroma Basics")
  chroma_basics()

  print_section("Search With Scores")
  similarity_search_with_scores()

  print_section("Metadata Filtering")
  metadata_filtering()

  print_section("Persist Chroma")
  persist_chroma()

  print_section("Vector Store as Retriever")
  vs_as_retriever(search_kwargs={"k": 3}, fetch_k=5)

  print_section("Exercise: Similarity Vector Store Setup")
  exercise_vector_store_setup()

  print_section("Exercise: MMR Vector Store Setup")
  exercise_vector_store_setup(search_type="mmr", fetch_k=5)
