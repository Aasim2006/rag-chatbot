"""
chat_state.py
--------------
Chat history with no Streamlit dependency, so it can be tested on its own.

Guarantees
- Every question gets a unique id and is answered at most once, so a Streamlit
  rerun (or an interrupted run) can never produce a duplicate answer.
- Submitting the same text again while that question is still waiting for its
  answer does not add a second copy.
- Each assistant message keeps its own status label and sources, so older
  answers still show them after a rerun.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    id: str
    role: str  # "user" or "assistant"
    content: str
    meta: dict = field(default_factory=dict)
    reply_to: Optional[str] = None


class ChatHistory:
    def __init__(self) -> None:
        self.messages: list = []
        self._answered: set = set()

    def add_user(self, text: Optional[str]) -> Optional[ChatMessage]:
        """Record a question. Returns None for empty input; returns the existing
        message (no duplicate) if the same question is still waiting for its answer."""
        text = (text or "").strip()
        if not text:
            return None
        waiting = self.pending()
        if waiting is not None and waiting.content == text:
            return waiting
        msg = ChatMessage(id=uuid.uuid4().hex, role="user", content=text)
        self.messages.append(msg)
        return msg

    def pending(self) -> Optional[ChatMessage]:
        """The oldest question that has no answer yet."""
        for m in self.messages:
            if m.role == "user" and m.id not in self._answered:
                return m
        return None

    def add_assistant(self, reply_to: str, content: str, meta: Optional[dict] = None) -> Optional[ChatMessage]:
        """Record the answer to a question. Returns None if that question was already
        answered (or doesn't exist), which makes duplicate answers impossible."""
        known = any(m.id == reply_to and m.role == "user" for m in self.messages)
        if not known or reply_to in self._answered:
            return None
        msg = ChatMessage(id=uuid.uuid4().hex, role="assistant", content=content,
                          meta=dict(meta or {}), reply_to=reply_to)
        self.messages.append(msg)
        self._answered.add(reply_to)
        return msg

    def clear(self) -> None:
        self.messages.clear()
        self._answered.clear()
