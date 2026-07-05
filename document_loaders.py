import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
  DirectoryLoader,
  PyPDFLoader,
  TextLoader,
  WebBaseLoader,
)
from langchain_core.documents import Document

load_dotenv()


def print_section(title):
  """Print a section header in blue."""
  print(f"\n\033[94m{'=' * 50}\n{title}\n{'=' * 50}\033[0m")


def clean_whitespace(text: str) -> str:
  """Collapse blank/whitespace-only lines left behind by raw HTML scraping."""
  lines = (line.strip() for line in text.splitlines())
  return "\n".join(line for line in lines if line)


# Text Loader example
def load_text_file():
  # Create a temporary text file for demonstration
  with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
    temp_file.write(
      b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader.\nIt contains multiple lines of text for testing purposes."
    )
    temp_file_path = temp_file.name

  try:
    # Load the text file using TextLoader
    loader = TextLoader(temp_file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    print(f"Metadata: {documents[0].metadata}")
    print(f"\nContent Preview:\n{documents[0].page_content[:100]}...")  # Print first 100 characters

    # Print the loaded documents
    for document in documents:
      print(f"\nDocument Content:\n{document.page_content}")  # Print first 100 characters
      # print(document)
      # print(document.page_content)
  finally:
    # Clean up the temporary file
    os.remove(temp_file_path)


# Web Loader example
def web_loader():
  web_url = "https://en.wikipedia.org/wiki/Web_scraping"
  loader = WebBaseLoader(
    web_url,
    bs_kwargs={"parse_only": None},
    bs_get_text_kwargs={"separator": " ", "strip": True},
  )
  documents = loader.load()

  for document in documents:
    document.page_content = clean_whitespace(document.page_content)

  print(f"\033[92mLoaded {len(documents)} document(s) from web {web_url}\033[0m\n")
  print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
  print(f"Content Length: {len(documents[0].page_content)} characters")
  print(f"\nContent Preview:\n{documents[0].page_content[:200]}...")


# Directory Lazy Loader example
def directory_loader():

  # Create temp directory with sample files
  with tempfile.TemporaryDirectory() as tmpdir:
    # Create sample files
    for i in range(5):
      path = Path(tmpdir) / f"doc_{i}.txt"
      path.write_text(f"This is document {i}. It contains sample content.")

    loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

    print(f"\033[92mInitialized Lazy Loader for Directory: {tmpdir}\033[0m\n")
    for doc in loader.lazy_load():
      print("Content:", doc.page_content)
      # print("Metadata:", doc.metadata["source"])
      print("Metadata:", doc.metadata)


# Document Structure example
def doc_structure():
  doc = Document(
    page_content="This is a sample document.",
    metadata={
      "source": "manual_creation.txt",
      "author": "Paulo",
      "length": 30,
      "tags": ["sample", "test"],
      "created_at": "2024-06-01",
    },
  )

  # print("\033[92mDocument Structure: \033[0m\n")
  print(f"Page Content Type: {type(doc.page_content)}")
  print(f"Page Content: {doc.page_content}")
  print(f"Metadata: {doc.metadata}")


# PDF Loader example
def pdf_loader(pdf_path: str):
  loader = PyPDFLoader(pdf_path)
  documents = loader.load()

  print(f"\033[92mLoaded {len(documents)} document(s) from PDF {pdf_path}\033[0m")
  for i, doc in enumerate(documents):
    print(f"\nPage Content `{i + 1}`: {doc.page_content[:100]}...")
    print(f"\nMetadata `{i + 1}`: {doc.metadata}")


if __name__ == "__main__":
  print_section("Text File Loader")
  load_text_file()
  print_section("Web Loader")
  web_loader()

  print_section("Directory Loader")
  directory_loader()

  print_section("Document Structure")
  doc_structure()

  file = Path("./docs/langchain_demo.pdf")
  print_section("PDF Loader File")
  pdf_loader(file)
