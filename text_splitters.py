"""
Text Splitters and Chunking Strategies
Optimizing document chunks for RAG
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
  Language,
  MarkdownHeaderTextSplitter,
  RecursiveCharacterTextSplitter,
)

load_dotenv()


def print_section(title):
  """Print a section header in blue."""
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
6. Healthcare diagnostics
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


# Recursive Character Text Splitter example
def recursive_splitter(text=SAMPLE_TEXT):
  chunk_size = 500
  chunk_overlap = 50
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
  )
  chunks = splitter.split_text(text)

  print(f"Type of Text: {type(text)}")
  print(f"Original length: {len(text)} chars")
  print(f"Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars\n")

  print(f"Number of chunks: {len(chunks)}")
  print(f"Split chunk sizes: {[len(c) for c in chunks]}")
  print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")


# Chunk Size Comparison example
def chunk_size_comparison(text=SAMPLE_TEXT):
  sizes = [200, 500, 1000]

  print(f"Type of Text: {type(text)}")
  print(f"SAMPLE TEXT size: {len(text)} chars\n")
  for size in sizes:
    splitter = RecursiveCharacterTextSplitter(
      chunk_size=size, chunk_overlap=size // 5
    )  # 20% overlap
    chunks = splitter.split_text(text)
    print(f" Size {size}: {len(chunks)} chunks")
    print(f" Split chunk sizes: {[len(c) for c in chunks]}")
    print(f" Total chars across chunks: {sum(len(c) for c in chunks)}\n")


# Overlap importance demonstration
def overlap_importance():
  text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text

  # without overlap
  splitter_no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

  # with overlap
  splitter_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

  chunks_no_overlap = splitter_no_overlap.split_text(text)
  chunks_with_overlap = splitter_overlap.split_text(text)

  print(f"Type of Text: {type(text)}")
  print(f"Original length: {len(text)} chars\n")

  print("Without overlap:")
  print(f"  Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
  print(f"  Chunk 2 start: {chunks_no_overlap[1][:20]}...")
  print(f"  Number of Chunks: {len(chunks_no_overlap)}")
  print(f"  Split chunk sizes: {[len(c) for c in chunks_no_overlap]}")

  print("\nWith overlap:")
  print(f"  Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
  print(f"  Chunk 2 start: {chunks_with_overlap[1][:20]}...")
  print(f"  Number of Chunks: {len(chunks_with_overlap)}")
  print(f"  Split chunk sizes: {[len(c) for c in chunks_with_overlap]}")


# Markdown Header Text Splitter example
def markdown_splitter(text=SAMPLE_TEXT):
  headers_to_consider = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
  ]
  splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
  chunks = splitter.split_text(text)

  print(f"Type of Text: {type(text)}")
  print(f"Markdown Splitter produced {len(chunks)} chunks.")
  for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(f" Metadata: {chunk.metadata}\n")
    print(f" Content: {chunk.page_content[:200]}...\n")


# Code Splitter example => Language.PYTHON
def code_splitter(text=SAMPLE_CODE):
  python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=500, chunk_overlap=50
  )
  chunks = python_splitter.split_text(text)

  print(f"Type of Text: {type(text)}")
  print(f"Code Length: {len(text)} chars")
  print(f"Code Splitter produced {len(chunks)} chunks.")
  for i, chunk in enumerate(chunks):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


# Document Splitter from PDF example
def document_splitter(filename="./docs/langchain_demo.pdf"):
  text = PyPDFLoader(filename).load()
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  # split the docs
  chunks = splitter.split_documents(text)

  print(f"Type of Text: {type(text)}")
  print(f"Loaded {len(text)} documents from PDF {filename}.")
  print(f"Split into {len(chunks)} chunks")
  print(f"\nFirst chunk metadata: {chunks[0].metadata}")
  print(f"First chunk content: {chunks[0].page_content[:200]}...")
  print(f"\nLast chunk metadata: {chunks[-1].metadata}")


if __name__ == "__main__":
  print_section("Recursive Character Text Splitter")
  recursive_splitter(SAMPLE_TEXT)

  print_section("Chunk Size Comparison")
  chunk_size_comparison(SAMPLE_TEXT)

  print_section("Overlap Importance Demonstration")
  overlap_importance()

  print_section("Markdown Splitter")
  markdown_splitter(SAMPLE_TEXT)

  print_section("Code Splitter")
  code_splitter(SAMPLE_CODE)

  print_section("Document Splitter from PDF")
  document_splitter("./docs/langchain_demo.pdf")
