"""
Dialog manager for history, context window, and session handling.
Covers multi-turn conversations, context management, and history trimming.
"""
import os
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - dependency is installed in deployment
    class ChatGoogleGenerativeAI:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "langchain_google_genai is not installed. Install project dependencies "
                "to enable Gemini-backed dialogue."
            )

from src.prompts.templates import (
    SYSTEM_PROMPT,
    PLAN_RECOMMENDATION_PROMPT,
    SELF_CORRECTION_PROMPT,
    HORECA_SPECIALIST_PROMPT,
    RETAIL_SPECIALIST_PROMPT,
)
from src.rag.retriever import TelecomRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# Intents where factual accuracy is critical — run self-correction for these.
_FACTUAL_INTENTS = {"plan_info", "billing", "connection", "porting"}
_ENABLE_SELF_CORRECTION = os.getenv("ENABLE_SELF_CORRECTION", "false").lower() == "true"

# Number of turns during which client type detection is attempted.
_CLIENT_TYPE_DETECTION_TURNS = 3
_INTENT_ALIASES = {
    "plan info": "plan_info",
    "plan_info": "plan_info",
    "technical issue": "technical_issue",
    "technical_issue": "technical_issue",
    "billing": "billing",
    "connection": "connection",
    "porting": "porting",
    "cancellation": "cancellation",
    "other": "other",
}


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

    def to_langchain_messages(self, max_messages: int = 10):
        """Convert history to LangChain message objects."""
        recent = self.messages[-max_messages:]
        messages = []
        for msg in recent:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            else:
                messages.append(SystemMessage(content=msg.content))
        return messages

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
        model: str = None,
        temperature: float = 0.3,
        max_history: int = 10,
    ):
        model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )
        self.control_client = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
        )
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
        """Determine the client intent with a local keyword heuristic."""
        text = message.lower()
        rules = [
            ("plan_info", [r"\bplan\b", r"\bprice\b", r"\bcost\b", r"\btariff\b", r"\bsubscription\b"]),
            ("technical_issue", [r"\bno internet\b", r"\binternet.*not working\b", r"\bnetwork\b", r"\bterminal\b", r"\berror\b"]),
            ("billing", [r"\bbill\b", r"\binvoice\b", r"\bcharged\b", r"\bpayment\b", r"\bdebt\b"]),
            ("connection", [r"\bconnect\b", r"\bsetup\b", r"\bactivate\b", r"\binstall\b", r"\bnew number\b", r"\b8-800\b"]),
            ("porting", [r"\btransfer\b", r"\bmnp\b", r"\bport\b", r"\bmove number\b"]),
            ("cancellation", [r"\bcancel\b", r"\bdisable\b", r"\bdisconnect\b"]),
        ]
        for intent, patterns in rules:
            if any(re.search(pattern, text) for pattern in patterns):
                return intent
        return "other"

    def _detect_client_type(self, message: str) -> Optional[str]:
        """
        Detect business type (horeca / retail) from a single message.
        Returns None if the type cannot be determined.
        """
        text = message.lower()
        horeca_keywords = ["restaurant", "cafe", "bar", "hotel", "hostel", "catering", "horeca"]
        retail_keywords = ["shop", "store", "boutique", "pharmacy", "supermarket", "retail", "mall"]
        if any(word in text for word in horeca_keywords):
            return "horeca"
        if any(word in text for word in retail_keywords):
            return "retail"
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
        if not _ENABLE_SELF_CORRECTION:
            return reply, False

        prompt = SELF_CORRECTION_PROMPT.format(response=reply, context=context)
        try:
            response = self.control_client.invoke(prompt)
            corrected = str(response.content).strip()
            was_corrected = corrected.startswith("[CORRECTED]")
            return corrected, was_corrected
        except Exception as e:
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
        messages = [SystemMessage(content=system_content)]
        messages.extend(session.to_langchain_messages(max_messages=self.max_history))
        messages.append(HumanMessage(content=user_message))

        # 6. Call the model
        try:
            response = self.client.invoke(messages)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                error_reply = (
                    "The free Gemini quota for this project was exceeded. "
                    "Please wait a bit and try again, or switch to a different Google project/key."
                )
                logger.error("Chat completion rate limited: %s", e)
                session.add_message("user", user_message, intent=intent)
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

        assistant_reply = str(response.content)
        usage = getattr(response, "usage_metadata", {}) or {}
        tokens_used = int(
            usage.get("total_tokens")
            or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            or max(1, len(assistant_reply.split()))
        )

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
