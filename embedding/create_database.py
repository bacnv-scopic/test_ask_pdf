import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import patch_sqlite3
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "embedding/data"


def load_pdf_from_path(path: str) -> list:
    """
    Load all pdf documents from a given path.
    """
    print(f"Loading PDF documents from {path}")
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    return docs


def slpit_text(docs: list) -> list:
    """
    Split the text into text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            ""
        ]
    )

    text_chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(text_chunks)} chunks.")
    return text_chunks


def save_to_chroma(chunks: list, chroma_path: str):
    # Clear out the database first.
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create a new DB from the documents.
    print(f"Saving {len(chunks)} chunks to {chroma_path}...")
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=chroma_path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")


def create_database():
    # load the pdf documents
    documents = load_pdf_from_path(DATA_PATH)

    # split the text content into chunks
    text_chunks = slpit_text(documents)

    # save the chunks to the chroma database
    save_to_chroma(text_chunks, CHROMA_PATH)


if __name__ == "__main__":
    create_database()
