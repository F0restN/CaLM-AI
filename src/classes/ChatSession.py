from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator


class MessageRole(str, Enum):
    """Enumeration for message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class BaseChatMessage(BaseModel):
    """Base chat message class."""

    role: MessageRole
    content: str


# Strategy Pattern for Message Formatting
class MessageFormatter(ABC):
    """Abstract base class for message formatting strategies."""

    @abstractmethod
    def format(self, messages: list[BaseChatMessage]) -> str:
        """Format messages according to specific strategy."""


class StandardFormatter(MessageFormatter):
    """Standard conversation formatter."""

    def format(self, messages: list[BaseChatMessage]) -> str:
        """Format conversation in standard format."""
        if not messages:
            return ""

        conversation = [
            f"{msg.role.value.upper()}: {msg.content}" for msg in messages
        ]
        return "\n".join(conversation)


# Main ChatSession Class
class ChatSessionFactory:
    """Factory for creating, managing and formatting chat sessions."""

    def __init__(
        self,
        messages: list[BaseChatMessage] | None = None,
        max_messages: int | None = None,
        formatter: MessageFormatter | None = None,
    ) -> None:
        """Initialize the chat session."""
        self._messages: list[BaseChatMessage] = messages or []
        self._formatter = formatter or StandardFormatter()
        self._max_messages: int = max_messages or 6

    @field_validator("messages", mode="after")
    @classmethod
    def validate_messages(cls) -> list[BaseChatMessage]:
        """Validate messages."""
        if cls._max_messages and len(cls._messages) > cls._max_messages:
            cls._messages = cls._messages[-cls._max_messages:]
        return cls._messages

    def get_latest_user_message(self) -> BaseChatMessage:
        """Get the latest user message."""
        for message in reversed(self._messages):
            if message.role == MessageRole.USER:
                return message
        return None

    def get_latest_assistant_message(self) -> BaseChatMessage:
        """Get the latest assistant message."""
        for message in reversed(self._messages):
            if message.role == MessageRole.ASSISTANT:
                return message
        return None

    def get_latest_conversation_pair(self) -> tuple[BaseChatMessage, BaseChatMessage]:
        """Get the latest conversation in sequence. Must start with a user message and followed by an assistant message."""
        user_message = self.get_latest_user_message()
        assistant_message = self.get_latest_assistant_message()
        if user_message and assistant_message:
            return user_message, assistant_message
        return None

    def get_formatted_conversation(self, attriute: Literal["messages", "latest_conversation_pair"] = "messages") -> str:
        """Get formatted conversation using the current formatter."""
        if attriute == "messages":
            return self._formatter.format(self._messages)
        if attriute == "latest_conversation_pair":
            return self._formatter.format(self.get_latest_conversation_pair())
        raise ValueError(f"Invalid attribute: {attriute}")

    def set_formatter(self, formatter: MessageFormatter) -> None:
        """Set a new formatting strategy."""
        self._formatter = formatter

    def __str__(self) -> str:
        """Get formatted conversation using the current formatter."""
        return self.get_formatted_conversation("messages")