# Model	              Dimensions	Cost per 1M tokens	Best For
# text-embedding-3-small	1536	    $0.02	            General use
# text-embedding-3-large	3072	    $0.13	                High accuracy
# text-embedding-ada-002	1536	    $0.10	                Legacy

from dotenv import load_dotenv

load_dotenv()


def print_section(title):
  """Print a section header in blue."""
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


print_section("Loading Embeddings")

# # 1 HuggingFace Dimensions 384
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(
#   model_name="sentence-transformers/all-MiniLM-L6-v2"
# )  # 384 dimensions

# model = "HuggingFace - " + embeddings.model_name  # Set the model name for printing

# # 2 Ollama with Qwen3-embedding:8b DIMENSIONS 4096
# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")  # 4096 dimensions
# model = "Ollama - " + embeddings.model  # Set the model name for printing

# 3 OpenAI Dimensions 1536
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = "OpenAI - " + embeddings.model  # Set the model name for printing

print(f"Embedding model: {model}")

print_section("Single Text Embedding")

# single text
text = "This is a sample text to be embedded."
embedding = embeddings.embed_query(text)
# print(f"Embedding for single text: {embedding}")

print(
  f"Length of single embedding: {len(embedding)}"
)  # Should print 1536 for text-embedding-3-small
print(f"First 10 dimensions of single embedding: {embedding[:10]}...")  # Print first 10 dimensions

print_section("Multiple Text Embeddings")

# multiple texts
embeds = embeddings.embed_documents(["This is the first document.", "This is the second document."])

print(f"Number of embeddings: {len(embeds)}\n")  # Should print 2
for i, embed in enumerate(embeds):
  print(f"Embedding {i + 1} length: {len(embed)}")  # Should print 1536 for text-embedding-3-small
  print(
    f"Embedding {i + 1} first 10 dimensions: {embed[:10]}...\n"
  )  # Print first 10 dimensions of each embedding
