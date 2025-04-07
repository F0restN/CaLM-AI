import datetime
from typing import List, Literal, Any
from pydantic import BaseModel, Field
from uuid import uuid4


"""
This is the class for the memory (LTM and STM)

Bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,are considered as global level memory which is LTM and stored in the database

Preferences, answer tone, language, etc., are considered as short term memory.

All memory will be generated and express as a sentences. Hence, for that sub class:
TODO: a sub class for sentence will be created. need to be able to convert attributes into sentences and vice versa.

NOTE: attributes should include: level (granularity), type (e.g. bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,), content (the sentence), source (e.g. user, system, care recipient, etc.), timestamp (when the sentence is created).

"""


class MemoryItem(BaseModel):
    id: int = Field(default_factory=lambda: uuid4().int, description="The unique identifier for the memory item")
    content: str = Field(description="Actual content of this memory attribute")
    level: str = Field(description="To which level of granularity this memory attribute belongs to")
    type: str = Field(description="attribute name")
    source: str = Field(description="where this memory comes from")
    timestamp: datetime = Field(description="when this memory is created")
    
    def __init__(self, id: int, content: str, level: str, type: str, source: str, timestamp: datetime):
        self.id = id
        self.content = content
        self.level = Literal["LTM", "STM"]
        self.type = type
        self.source = source
        self.timestamp = timestamp
    
    @classmethod
    def convert_to_sentence(cls):
        pass
    
    @classmethod
    def convert_to_attributes(cls):
        pass
        
    def __str__(self):
        return f"MemoryItem(id={self.id}, content={self.content}, level={self.level}, type={self.type}, source={self.source}, timestamp={self.timestamp})"

    def __repr__(self):
        return self.__str__()


class Memory(BaseModel):
    id: int
    relationship: List[str]
    
    created_at: datetime
    updated_at: datetime

    def __init__(self, id: int, content: str, created_at: datetime, updated_at: datetime):
        self.id = id
        self.content = content
        self.created_at = created_at
        self.updated_at = updated_at

    def __str__(self):
        return f"Memory(id={self.id}, content={self.content}, created_at={self.created_at}, updated_at={self.updated_at})"

    def __repr__(self):
        return self.__str__()


class MetadataString(BaseModel):
    """
    A class representing a string with metadata.
    """
    content: str = Field(description="The actual string content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the string")
    created_at: datetime = Field(default_factory=datetime.now, description="When the string was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When the string was last updated")
    
    def __str__(self) -> str:
        return self.content
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update a specific metadata field.
        
        Args:
            key: The metadata key to update
            value: The new value for the metadata
        """
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a specific metadata value.
        
        Args:
            key: The metadata key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The metadata value or default if not found
        """
        return self.metadata.get(key, default)
    
    def to_dict(self) -> dict:
        """
        Convert the MetadataString to a dictionary.
        
        Returns:
            A dictionary representation of the MetadataString
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MetadataString":
        """
        Create a MetadataString from a dictionary.
        
        Args:
            data: Dictionary containing the string data
            
        Returns:
            A new MetadataString instance
        """
        return cls(**data)
