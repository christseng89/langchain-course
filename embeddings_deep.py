import numpy as np
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
print(f"\033[92m\nEmbedding model: {embeddings_model.model}\033[0m")

# Documents
DOCS = [
  "Python is a programming language",
  "JavaScript is used for web development",
  "Machine learning enables AI applications",
  "Deep learning uses neural networks",
  "Cats are popular pets",
]


def print_section(title):
  """Print a section header in blue."""
  print(f"\n\033[94m{'=' * 60}\n{title}\n{'=' * 60}\033[0m")


def basic_embeddings():
  print_section("BASIC EMBEDDINGS")

  # single text
  text = "What is Machine Learning?"
  single_embedding = embeddings_model.embed_query(text)
  print(f"Vector dimensions: {len(single_embedding)}")
  print(f"First 5 values: {single_embedding[:5]}")
  print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
  print_section("BATCH EMBEDDINGS")

  text = [
    "What is Machine Learning?",
    "Explain the concept of overfitting in ML.",
    "How does a neural network work?",
  ]

  batch_embedding = embeddings_model.embed_documents(text)
  for i, emb in enumerate(batch_embedding):
    print(f"Text {i + 1}")
    print(f" Vector dimensions: {len(emb)}")
    print(f" First 5 values: {emb[:5]}")
    print(f" Vector norm: {np.linalg.norm(emb):.4f}\n")
    # 把每個維度的值平方、加總、再開根號。
    # OpenAI 的 embedding 向量通常已經正規化（normalized）， norm 應該接近 1.0。


def similarity_search():
  print_section("SIMILARITY SEARCH")

  query = "What programming languages exist?"

  # embed documents and query
  doc_vector = embeddings_model.embed_documents(DOCS)
  query_vector = embeddings_model.embed_query(query)

  # compute cosine similarities
  def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

  similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

  # rank documents by similarity
  ranked_docs = sorted(zip(DOCS, similarities), key=lambda x: x[1], reverse=True)

  print(f"Query: {query}\n")
  print("Ranked by similarity:")
  for doc, score in ranked_docs:
    print(f"  {score:.4f}: {doc}")


# Caching ---
def embedding_caching():
  print_section("EMBEDDING CACHING")

  import tempfile

  from langchain_classic.embeddings.cache import CacheBackedEmbeddings
  from langchain_classic.storage import LocalFileStore

  with tempfile.TemporaryDirectory() as tempdir:
    local_store = LocalFileStore(root_path=tempdir)

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
      underlying_embeddings=embeddings_model,
      document_embedding_cache=local_store,
      namespace="exercise",
      key_encoder="sha256",
    )

    text = "What is Reinforcement Learning?"

    # First call - hits API
    print("First call (API):")
    vectors1 = embeddings_model.embed_documents([text])
    print(f"  Embedded {len(vectors1)} documents")

    # Second call - from cache
    print("\nSecond call (Cache):")
    vectors2 = cached_embeddings.embed_documents([text])
    print(f"  Embedded {len(vectors2)} documents")

    # Verify same results
    print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")


if __name__ == "__main__":
  basic_embeddings()
  batch_embeddings()
  similarity_search()
  embedding_caching()
