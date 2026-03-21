"""
RAG module for document indexing and semantic search.
Covers chunking, embeddings, vector database storage, and retrieval.
"""
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class TelecomIndexer:
    """Index knowledge base documents into a vector database."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

    def index_directory(self, data_dir: str) -> int:
        """Index all `.txt` files from a directory."""
        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()

        chunks = self.splitter.split_documents(documents)

        # Add metadata so the original source can be tracked
        for chunk in chunks:
            chunk.metadata["source_file"] = Path(chunk.metadata.get("source", "")).name

        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        print(f"[RAG] Indexed {len(documents)} file(s), {len(chunks)} chunk(s)")
        return len(chunks)

    def index_texts(self, texts: List[str], metadatas: List[dict] = None) -> int:
        """Index a list of texts directly."""
        documents = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        chunks = self.splitter.split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return len(chunks)


class TelecomRetriever:
    """Retrieve relevant passages for a query."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

    def get_context(self, query: str, k: int = 4) -> str:
        """Return prompt context as a string."""
        docs = self.vectordb.similarity_search(query, k=k)
        if not docs:
            return "No relevant information was found in the knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "FAQ")
            parts.append(f"[Excerpt {i} from {source}]\n{doc.page_content}")

        return "\n\n".join(parts)

    def get_docs(self, query: str, k: int = 4) -> List[Document]:
        """Return a list of `Document` objects."""
        return self.vectordb.similarity_search(query, k=k)

    def get_context_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Return `(document, score)` pairs for quality analysis."""
        return self.vectordb.similarity_search_with_score(query, k=k)

    def is_ready(self) -> bool:
        """Check whether the database has already been indexed."""
        try:
            count = self.vectordb._collection.count()
            return count > 0
        except Exception:
            return False
