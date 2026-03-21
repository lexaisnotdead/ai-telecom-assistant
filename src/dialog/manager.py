"""
Dialog manager for history, context window, and session handling.
Covers multi-turn conversations, context management, and history trimming.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from openai import OpenAI

from src.prompts.templates import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from src.rag.retriever import TelecomRetriever


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    client_type: Optional[str] = None  # "horeca" | "retail" | "microbusiness"
    created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, intent: str = None):
        self.messages.append(Message(role=role, content=content, intent=intent))

    def get_history_text(self, last_n: int = 6) -> str:
        """Format chat history for the prompt."""
        recent = self.messages[-last_n:]
        if not recent:
            return "Start of conversation."
        lines = []
        for msg in recent:
            prefix = "Client" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def to_openai_format(self, max_messages: int = 10) -> List[dict]:
        """Convert history to the OpenAI API message format."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]


class DialogManager:
    """
    Manages dialogue sessions.
    Uses RAG to enrich the context for each reply.
    """

    def __init__(
        self,
        retriever: TelecomRetriever,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.3,
        max_history: int = 10,
    ):
        self.client = OpenAI()
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def classify_intent(self, message: str) -> str:
        """Determine the client intent with few-shot classification."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip()

    def chat(self, session_id: str, user_message: str) -> dict:
        """
        Main dialogue method.
        Returns a dictionary with the reply and metadata.
        """
        session = self.get_or_create_session(session_id)

        # 1. Detect intent
        intent = self.classify_intent(user_message)

        # 2. Retrieve relevant context from RAG
        context = self.retriever.get_context(user_message, k=4)

        # 3. Gather dialogue history
        chat_history = session.get_history_text(last_n=6)

        # 4. Build the system prompt with context
        system_content = SYSTEM_PROMPT.format(
            chat_history=chat_history,
            context=context,
        )

        # 5. Build API messages
        messages = [{"role": "system", "content": system_content}]

        # Add dialogue history (the first instruction is already in system)
        history = session.to_openai_format(max_messages=self.max_history)
        messages.extend(history)

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        # 6. Call the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=600,
        )

        assistant_reply = response.choices[0].message.content

        # 7. Save the exchange into session history
        session.add_message("user", user_message, intent=intent)
        session.add_message("assistant", assistant_reply)

        return {
            "reply": assistant_reply,
            "intent": intent,
            "context_used": context[:200] + "..." if len(context) > 200 else context,
            "session_id": session_id,
            "turn": len(session.messages) // 2,
            "tokens_used": response.usage.total_tokens,
        }

    def reset_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
