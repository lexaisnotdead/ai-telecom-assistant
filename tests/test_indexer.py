import pytest

import src.rag.retriever as retriever_module


def test_prepare_persist_dir_blocks_duplicate_index(monkeypatch, tmp_path):
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [[0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 0.0]

    monkeypatch.setattr(
        retriever_module,
        "SentenceTransformerEmbeddings",
        lambda model_name=None: DummyEmbeddings(),
    )
    indexer = retriever_module.TelecomIndexer(persist_dir=str(tmp_path / "db"))
    monkeypatch.setattr(indexer, "_has_existing_index", lambda: True)

    with pytest.raises(ValueError, match="overwrite=True"):
        indexer._prepare_persist_dir(overwrite=False)
