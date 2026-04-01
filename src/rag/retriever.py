"""
RAG module for document indexing and semantic search.
Covers chunking, embeddings, vector database storage, and retrieval.
"""
import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older environments
    from langchain_community.vectorstores import Chroma

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    SentenceTransformer = None


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """
    Lightweight local embedding wrapper for Chroma.

    Uses a multilingual sentence-transformers model so the project does not
    depend on external embedding APIs.
    """

    def __init__(
        self,
        model_name: str = None,
    ):
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.model_name = model_name
        self.dimensions = 384
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        if self.model is None:
            logger.warning(
                "sentence_transformers is unavailable or the model could not be loaded; "
                "using a lightweight fallback embedding implementation."
            )

    def _fallback_encode(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"\w+", text.lower())
            if not tokens:
                vectors.append(vector)
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            return self._fallback_encode(texts)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self.model is None:
            return self._fallback_encode([text])[0]
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

class TelecomIndexer:
    """Index knowledge base documents into a vector database."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = SentenceTransformerEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

    def _has_existing_index(self) -> bool:
        """Check whether the persist directory already contains indexed documents."""
        try:
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            return vectordb._collection.count() > 0
        except Exception:
            return False

    def _prepare_persist_dir(self, overwrite: bool) -> None:
        """Prepare the persist directory before indexing."""
        persist_path = Path(self.persist_dir)

        if overwrite and persist_path.exists():
            shutil.rmtree(persist_path)
            return

        if self._has_existing_index():
            raise ValueError(
                f"Index already exists at '{self.persist_dir}'. "
                "Use overwrite=True to rebuild it."
            )

    def index_directory(self, data_dir: str, overwrite: bool = False) -> int:
        """Index all `.txt` files from a directory."""
        self._prepare_persist_dir(overwrite=overwrite)
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

    def index_texts(
        self,
        texts: List[str],
        metadatas: List[dict] = None,
        overwrite: bool = False,
    ) -> int:
        """Index a list of texts directly."""
        self._prepare_persist_dir(overwrite=overwrite)
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

    def append_texts(
        self,
        texts: List[str],
        metadatas: List[dict] = None,
    ) -> int:
        """Append new texts to an existing or fresh index."""
        documents = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        chunks = self.splitter.split_documents(documents)

        try:
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            if vectordb._collection.count() == 0:
                raise ValueError
            vectordb.add_documents(chunks)
        except Exception:
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        return len(chunks)


class TelecomRetriever:
    """Retrieve relevant passages for a query."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = SentenceTransformerEmbeddings()
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
