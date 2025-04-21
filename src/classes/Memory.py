from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.GLOBAL import CATEGORIES

"""
This is the class for the memory (LTM and STM)

Bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,
are considered as global level memory which is LTM and stored in the database

Preferences, answer tone, language, etc., are considered as short term memory.

All memory will be generated and express as a sentences. Hence, for that sub class:
TODO: finalize the class. Including documentation and attributes definition.

NOTE: attributes should include: level (granularity), type (e.g. bio, job, social relationship, relationship with care
recipient,topics if interest to user etc.,),content (the sentence), source (e.g. user, system, care recipient, etc.),
timestamp (when the sentence is created).

NOTE:
category, type, content in sentence.
e.g. "The user's {job} is a {nurse}" / "[CATEGORY: {ALZ}] The user's {care recipient} is {dad}"
level, type, source, timestamp in attributes.
"""


class BaseMemory(BaseModel):
    """Basic memory attribute for AI function calling."""

    content: str = Field(description="Actual content of this memory attribute")
    level: Literal["LTM", "STM"] = Field(description="To which level of granularity this memory attribute belongs to")
    category: str = Field(description="category of this memory attribute")
    type: str = Field(description="attribute name")
    topic: list[str] = Field(description="3 words topics that most representative to the content of this memory")

class MemoryItem(BaseMemory):
    """Memory item for AI function calling."""

    id: int = Field(default_factory=lambda: uuid4().int, description="The unique identifier for the memory item")
    user_id: str = Field(..., description="The unique identifier for the user")
    source: str = Field(description="where this memory comes from")
    timestamp: datetime = Field(
        description="when this memory is created, must be string",
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
    )
    metadata: dict = Field(
        default_factory=lambda data: {
            "id": data["id"],
            "user_id": data["user_id"],
            # Content provided in document pageContent attribute
            "level": data["level"],
            "category": data["category"],
            "type": data["type"],
            "topic": data["topic"],
            "source": data["source"],
            "timestamp": data["timestamp"].isoformat() if isinstance(data["timestamp"], datetime) else data["timestamp"],
        }, description="Additional metadata about the memory item",
    )

    def convert_to_sentence(self, categories: list[str] = CATEGORIES) -> str:
        """Convert the memory item to a sentence."""
        if self.category in categories:
            return f"[CATEGORY: {self.category}] The user's {self.type} is {self.content}"
        return f"The user's {self.type} is {self.content}"

    # def convert_to_attributes(self) -> None:
    #     """Convert the memory item to attributes."""
    #     pass

    def __str__(self) -> str:
        return f"MemoryItem(id={self.id}, content={self.content}, level={self.level}, type={self.type}, source={self.source}, timestamp={self.timestamp}, topic={self.topic})"

    def __repr__(self) -> str:
        return self.__str__()


class Memory(BaseModel):
    """Memory for AI function calling."""

    id: int
    user_id: int = Field(description="The unique identifier for the user")
    user_profile: list[MemoryItem]
    created_at: datetime
    updated_at: datetime

    def __init__(self, id: int, user_profile: list[MemoryItem], created_at: datetime, updated_at: datetime) -> None:
        """Initialize the memory."""
        self.id = id
        self.user_profile = user_profile
        self.created_at = created_at
        self.updated_at = updated_at

    def __str__(self) -> str:
        return f"Memory(id={self.id}, content={self.content}, created_at={self.created_at}, updated_at={self.updated_at})"

    def __repr__(self) -> str:
        return self.__str__()
