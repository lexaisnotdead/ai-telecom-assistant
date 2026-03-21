import pytest

import src.rag.retriever as retriever_module


def test_prepare_persist_dir_blocks_duplicate_index(monkeypatch, tmp_path):
    class DummyEmbeddings:
        pass

    monkeypatch.setattr(
        retriever_module,
        "OpenAIEmbeddings",
        lambda model: DummyEmbeddings(),
    )
    indexer = retriever_module.TelecomIndexer(persist_dir=str(tmp_path / "db"))
    monkeypatch.setattr(indexer, "_has_existing_index", lambda: True)

    with pytest.raises(ValueError, match="overwrite=True"):
        indexer._prepare_persist_dir(overwrite=False)
