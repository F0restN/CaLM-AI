from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ValidationInfo, field_validator


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


class ChatSessionFactory(BaseModel):
    """Factory for creating, managing and formatting chat sessions."""

    max_messages: int = 6
    formatter: MessageFormatter = StandardFormatter()
    messages: list[BaseChatMessage] = []

    class Config:
        arbitrary_types_allowed = True

    @field_validator("messages", mode="after")
    @classmethod
    def validate_messages(cls, v: list[BaseChatMessage], info: ValidationInfo) -> list[BaseChatMessage]:
        """Validate messages and enforce max_messages limit."""
        max_msgs = info.data["max_messages"]
        if max_msgs and len(v) > max_msgs:
            return v[-max_msgs:]
        return v

    def get_latest_user_message(self, *, last_n: int = 1) -> BaseChatMessage:
        """Get the latest user message. If last_n is greater than 1, return the last n user messages."""
        user_messages = [msg for msg in reversed(self.messages) if msg.role == MessageRole.USER]
        if len(user_messages) < last_n:
            return user_messages[0]
        return user_messages[-last_n]

    def get_latest_assistant_message(self) -> BaseChatMessage:
        """Get the latest assistant message."""
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message
        return None

    def get_latest_conversation_pair(self) -> tuple[BaseChatMessage, BaseChatMessage]:
        """Get the latest conversation in sequence. Must start with a user message and followed by an assistant message."""
        user_message = self.get_latest_user_message(last_n=2)
        assistant_message = self.get_latest_assistant_message()
        if user_message and assistant_message:
            return user_message, assistant_message
        return None

    def get_formatted_conversation(self, attribute: Literal["messages", "latest_conversation_pair"] = "messages") -> str:
        """Get formatted conversation using the current formatter."""
        if attribute == "messages":
            return self.formatter.format(self.messages)
        if attribute == "latest_conversation_pair":
            conversation_pair = self.get_latest_conversation_pair()
            if conversation_pair:
                return self.formatter.format(list(conversation_pair))
            return ""
        raise ValueError(f"Invalid attribute: {attribute}")

    def set_formatter(self, formatter: MessageFormatter) -> None:
        """Set a new formatting strategy."""
        self.formatter = formatter

    def __str__(self) -> str:
        """Get formatted conversation using the current formatter."""
        return self.get_formatted_conversation("messages")

