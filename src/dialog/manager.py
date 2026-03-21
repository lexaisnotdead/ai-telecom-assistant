"""
Dialog manager for history, context window, and session handling.
Covers multi-turn conversations, context management, and history trimming.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from openai import OpenAI, OpenAIError

from src.prompts.templates import (
    SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    CLIENT_TYPE_DETECTION_PROMPT,
    PLAN_RECOMMENDATION_PROMPT,
    SELF_CORRECTION_PROMPT,
    HORECA_SPECIALIST_PROMPT,
    RETAIL_SPECIALIST_PROMPT,
)
from src.rag.retriever import TelecomRetriever

logger = logging.getLogger(__name__)

# Intents where factual accuracy is critical — run self-correction for these.
_FACTUAL_INTENTS = {"plan_info", "billing", "connection", "porting"}

# Number of turns during which client type detection is attempted.
_CLIENT_TYPE_DETECTION_TURNS = 3


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
    client_type: Optional[str] = None  # "horeca" | "retail" | None
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

    @property
    def turn_count(self) -> int:
        return len(self.messages) // 2


class DialogManager:
    """
    Manages dialogue sessions.
    Uses RAG to enrich the context for each reply.

    Routing logic per turn:
      - First _CLIENT_TYPE_DETECTION_TURNS turns: attempts to detect client type
        (horeca / retail) and stores it on the session.
      - intent == "plan_info"           → PLAN_RECOMMENDATION_PROMPT (chain-of-thought)
      - session.client_type == "horeca" → HORECA_SPECIALIST_PROMPT
      - session.client_type == "retail" → RETAIL_SPECIALIST_PROMPT
      - otherwise                       → SYSTEM_PROMPT (generic)

    After generating a reply for factual intents (plan_info, billing,
    connection, porting), a self-correction pass is run via SELF_CORRECTION_PROMPT
    to catch hallucinated numbers or contradictions with the knowledge base.
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

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def reset_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def classify_intent(self, message: str) -> str:
        """Determine the client intent with few-shot classification."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logger.error("Intent classification failed: %s", e)
            return "other"

    def _detect_client_type(self, message: str) -> Optional[str]:
        """
        Detect business type (horeca / retail) from a single message.
        Returns None if the type cannot be determined.
        """
        prompt = CLIENT_TYPE_DETECTION_PROMPT.format(message=message)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().lower()
            return result if result in ("horeca", "retail") else None
        except OpenAIError as e:
            logger.error("Client type detection failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # System prompt routing
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        session: Session,
        context: str,
        user_message: str,
        intent: str,
    ) -> str:
        """
        Choose and format the system prompt based on intent and client type.

        Priority:
          1. plan_info intent → chain-of-thought plan recommendation
          2. Known client type (horeca / retail) → industry specialist
          3. Generic fallback
        """
        if intent == "plan_info":
            return PLAN_RECOMMENDATION_PROMPT.format(
                plans_info=context,
                client_request=user_message,
            )
        if session.client_type == "horeca":
            return HORECA_SPECIALIST_PROMPT.format(
                context=context,
                question=user_message,
            )
        if session.client_type == "retail":
            return RETAIL_SPECIALIST_PROMPT.format(
                context=context,
                question=user_message,
            )
        return SYSTEM_PROMPT.format(context=context)

    # ------------------------------------------------------------------
    # Self-correction
    # ------------------------------------------------------------------

    def _self_correct(self, reply: str, context: str) -> tuple[str, bool]:
        """
        Run a self-correction pass on the generated reply.

        Returns (final_reply, was_corrected).
        Falls back to the original reply on any API error.
        """
        prompt = SELF_CORRECTION_PROMPT.format(response=reply, context=context)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=700,
            )
            corrected = response.choices[0].message.content.strip()
            was_corrected = corrected.startswith("[CORRECTED]")
            return corrected, was_corrected
        except OpenAIError as e:
            logger.error("Self-correction failed: %s", e)
            return reply, False

    # ------------------------------------------------------------------
    # Main dialogue entry point
    # ------------------------------------------------------------------

    def chat(self, session_id: str, user_message: str) -> dict:
        """
        Main dialogue method.
        Returns a dictionary with the reply and metadata.
        """
        session = self.get_or_create_session(session_id)

        # 1. Detect intent
        intent = self.classify_intent(user_message)

        # 2. Try to detect client type on early turns (only while still unknown)
        if session.client_type is None and session.turn_count < _CLIENT_TYPE_DETECTION_TURNS:
            detected = self._detect_client_type(user_message)
            if detected:
                session.client_type = detected
                logger.info(
                    "Session %s: client type set to '%s'", session_id, detected
                )

        # 3. Retrieve relevant context from RAG
        context = self.retriever.get_context(user_message, k=4)

        # 4. Build the system prompt based on intent and client type
        system_content = self._build_system_prompt(session, context, user_message, intent)

        # 5. Assemble API messages: system + history + current user turn
        messages = [{"role": "system", "content": system_content}]
        messages.extend(session.to_openai_format(max_messages=self.max_history))
        messages.append({"role": "user", "content": user_message})

        # 6. Call the model
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=600,
            )
        except OpenAIError as e:
            logger.error("Chat completion failed: %s", e)
            session.add_message("user", user_message, intent=intent)
            error_reply = "I'm sorry, I'm temporarily unavailable. Please try again in a moment."
            session.add_message("assistant", error_reply)
            return {
                "reply": error_reply,
                "intent": intent,
                "client_type": session.client_type,
                "corrected": False,
                "context_used": "",
                "session_id": session_id,
                "turn": session.turn_count,
                "tokens_used": 0,
            }

        assistant_reply = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # 7. Self-correction for factual intents
        corrected = False
        if intent in _FACTUAL_INTENTS:
            assistant_reply, corrected = self._self_correct(assistant_reply, context)

        # 8. Persist the exchange
        session.add_message("user", user_message, intent=intent)
        session.add_message("assistant", assistant_reply)

        return {
            "reply": assistant_reply,
            "intent": intent,
            "client_type": session.client_type,
            "corrected": corrected,
            "context_used": context[:200] + "..." if len(context) > 200 else context,
            "session_id": session_id,
            "turn": session.turn_count,
            "tokens_used": tokens_used,
        }