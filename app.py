"""
Telecom Assistant web app for Hugging Face Spaces.

Run locally with:
    source .venv/bin/activate
    python app.py
"""
import os
import uuid
from pathlib import Path

import gradio as gr

from main import ensure_indexed
from src.agents.react_agent import run_agent
from src.dialog.manager import DialogManager
from src.rag.retriever import TelecomIndexer, TelecomRetriever


PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate")
CSS = """
#app-shell {
    max-width: 1180px;
    margin: 0 auto;
}
.app-title {
    margin-bottom: 0.25rem;
}
.app-subtitle {
    margin-top: 0;
    color: var(--body-text-color-secondary);
}
.chat-panel {
    min-height: 74vh;
}
.compact-note {
    color: var(--body-text-color-secondary);
    font-size: 0.95rem;
    margin-top: 0.25rem;
}
.admin-box {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 1rem;
    background: var(--background-fill-secondary);
}
"""


retriever = ensure_indexed(persist_dir=PERSIST_DIR, data_dir=DATA_DIR)
dialog_manager = None


def get_dialog_manager():
    """Create a Gemini-backed dialog manager only when credentials exist."""
    global dialog_manager
    if dialog_manager is not None:
        return dialog_manager
    if not GOOGLE_API_KEY:
        return None
    dialog_manager = DialogManager(retriever=retriever)
    return dialog_manager


def refresh_runtime():
    """Recreate the retriever and dialog manager after the index changes."""
    global retriever, dialog_manager
    retriever = TelecomRetriever(persist_dir=PERSIST_DIR)
    dialog_manager = DialogManager(retriever=retriever) if GOOGLE_API_KEY else None


def _session_key(prefix: str, request: gr.Request | None) -> str:
    if request is None or not getattr(request, "session_hash", None):
        return f"{prefix}-{uuid.uuid4().hex}"
    return f"{prefix}-{request.session_hash}"


def normalize_history(history):
    """Normalize history into Gradio-compatible message dictionaries."""
    if not history:
        return []

    if isinstance(history[0], dict) and "role" in history[0] and "content" in history[0]:
        return history

    normalized = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                normalized.append({"role": "user", "content": user_text})
            if assistant_text:
                normalized.append({"role": "assistant", "content": assistant_text})
    return normalized


def chat_reply(message: str, history, request: gr.Request | None = None):
    """Generate the next assistant reply in chat mode."""
    message = (message or "").strip()
    if not message:
        return normalize_history(history)

    if not GOOGLE_API_KEY:
        history = normalize_history(history)
        history.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "Gemini is not configured for this Space yet. "
                        "Please add GOOGLE_API_KEY in the Space secrets."
                    ),
                },
            ]
        )
        return history

    session_id = _session_key("chat", request)
    manager = get_dialog_manager()
    if manager is None:
        history = normalize_history(history)
        history.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": "Gemini is not configured for this Space yet.",
                },
            ]
        )
        return history

    result = manager.chat(session_id=session_id, user_message=message)
    history = normalize_history(history)
    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["reply"]},
        ]
    )
    return history


def agent_reply(message: str, history, request: gr.Request | None = None):
    """Generate the next assistant reply in agent mode."""
    message = (message or "").strip()
    if not message:
        return normalize_history(history)

    if not GOOGLE_API_KEY:
        history = normalize_history(history)
        history.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "Gemini is not configured for this Space yet. "
                        "Please add GOOGLE_API_KEY in the Space secrets."
                    ),
                },
            ]
        )
        return history

    answer = run_agent(retriever=retriever, question=message)
    history = normalize_history(history)
    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
    )
    return history


def reindex_knowledge_base():
    """Force a full rebuild of the Chroma index."""
    indexer = TelecomIndexer(persist_dir=PERSIST_DIR)
    count = indexer.index_directory(DATA_DIR, overwrite=True)
    refresh_runtime()
    return f"Reindexed {count} chunk(s) into `{PERSIST_DIR}`."


def add_admin_documents(uploaded_files, pasted_text, document_title):
    """Add new documents from uploaded files or pasted text."""
    files = uploaded_files or []
    if isinstance(files, (str, Path)):
        files = [files]

    pasted_text = (pasted_text or "").strip()
    document_title = (document_title or "").strip()

    texts = []
    metadatas = []

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        target_name = f"admin_{path.name}"
        (data_dir / target_name).write_text(content, encoding="utf-8")
        texts.append(content)
        metadatas.append({"source_file": target_name, "source": target_name})

    if pasted_text:
        target_name = f"admin_manual_{uuid.uuid4().hex}.txt"
        (data_dir / target_name).write_text(pasted_text, encoding="utf-8")
        texts.append(pasted_text)
        metadatas.append(
            {"source_file": target_name, "source": target_name, "title": document_title}
        )

    if not texts:
        return "No valid documents were provided."

    indexer = TelecomIndexer(persist_dir=PERSIST_DIR)
    chunks = indexer.append_texts(texts=texts, metadatas=metadatas)
    refresh_runtime()
    return f"Added {len(texts)} document(s) and indexed {chunks} chunk(s)."


def unlock_admin(password: str):
    """Reveal admin controls when the password matches."""
    if not ADMIN_PASSWORD:
        return gr.update(visible=True), "Admin mode is open."
    if password == ADMIN_PASSWORD:
        return gr.update(visible=True), "Admin unlocked."
    return gr.update(visible=False), "Wrong password."


with gr.Blocks(title="Telecom Assistant", fill_height=True) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.Markdown("# Telecom Assistant", elem_classes=["app-title"])
        gr.Markdown(
            "A compact support demo with two modes: customer chat and tool-using agent.",
            elem_classes=["app-subtitle"],
        )

        with gr.Tabs():
            with gr.TabItem("Chat"):
                gr.Markdown("Use this mode for customer support questions.")
                gr.Markdown(
                    "Try: plans for a cafe, billing questions, connection setup, or technical issues.",
                    elem_classes=["compact-note"],
                )
                chat_box = gr.Chatbot(label="Conversation", height=560)
                chat_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask about plans, billing, connection, or technical issues...",
                    lines=2,
                )
                with gr.Row():
                    chat_send = gr.Button("Send", variant="primary")
                    chat_clear = gr.Button("Clear")

                chat_send.click(
                    chat_reply,
                    inputs=[chat_input, chat_box],
                    outputs=[chat_box],
                ).then(lambda: "", outputs=chat_input)

                chat_input.submit(
                    chat_reply,
                    inputs=[chat_input, chat_box],
                    outputs=[chat_box],
                ).then(lambda: "", outputs=chat_input)

                chat_clear.click(lambda: [], outputs=[chat_box])

            with gr.TabItem("Agent"):
                gr.Markdown("Use this mode when you want the assistant to call tools.")
                gr.Markdown(
                    "Try: cost estimates, compatibility checks, or support contacts.",
                    elem_classes=["compact-note"],
                )
                agent_box = gr.Chatbot(label="Agent Conversation", height=560)
                agent_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask the agent to search knowledge, calculate costs, or check compatibility...",
                    lines=2,
                )
                with gr.Row():
                    agent_send = gr.Button("Send", variant="primary")
                    agent_clear = gr.Button("Clear")

                agent_send.click(
                    agent_reply,
                    inputs=[agent_input, agent_box],
                    outputs=[agent_box],
                ).then(lambda: "", outputs=agent_input)

                agent_input.submit(
                    agent_reply,
                    inputs=[agent_input, agent_box],
                    outputs=[agent_box],
                ).then(lambda: "", outputs=agent_input)

                agent_clear.click(lambda: [], outputs=[agent_box])

            with gr.TabItem("Admin"):
                gr.Markdown("Upload new knowledge base documents here.")
                gr.Markdown(
                    "Drag and drop `.txt` files or paste new text, then add it to the index.",
                    elem_classes=["compact-note"],
                )

                admin_password = gr.Textbox(
                    label="Admin password",
                    type="password",
                    placeholder="Enter admin password",
                )
                unlock_button = gr.Button("Unlock admin", variant="primary")
                unlock_status = gr.Markdown("")

                with gr.Column(visible=not bool(ADMIN_PASSWORD), elem_classes=["admin-box"]) as admin_controls:
                    admin_title = gr.Textbox(
                        label="Document title",
                        placeholder="Optional title for pasted content",
                    )
                    admin_files = gr.File(
                        label="Drop text files here",
                        file_count="multiple",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    admin_text = gr.TextArea(
                        label="Paste document text",
                        placeholder="Paste FAQ, pricing notes, support rules, or any other knowledge base content here...",
                        lines=10,
                    )
                    with gr.Row():
                        admin_add = gr.Button("Add documents", variant="primary")
                        admin_reindex = gr.Button("Reindex all docs")
                    admin_status = gr.Markdown("")

                unlock_button.click(
                    unlock_admin,
                    inputs=[admin_password],
                    outputs=[admin_controls, unlock_status],
                )

                admin_add.click(
                    add_admin_documents,
                    inputs=[admin_files, admin_text, admin_title],
                    outputs=[admin_status],
                )

                admin_reindex.click(
                    reindex_knowledge_base,
                    outputs=[admin_status],
                )


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
