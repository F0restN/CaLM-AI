from datetime import datetime
from typing import List, Literal
from pydantic import BaseModel, Field
from uuid import uuid4


"""
This is the class for the memory (LTM and STM)

Bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,are considered as global level memory which is LTM and stored in the database

Preferences, answer tone, language, etc., are considered as short term memory.

All memory will be generated and express as a sentences. Hence, for that sub class:
TODO: a sub class for sentence will be created. need to be able to convert attributes into sentences and vice versa.

NOTE: attributes should include: level (granularity), type (e.g. bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,), content (the sentence), source (e.g. user, system, care recipient, etc.), timestamp (when the sentence is created).

NOTE:
category, type, content in sentence. 
e.g. "The user's {job} is a {nurse}" / "[CATEGORY: {ALZ}] The user's {care recipient} is {dad}"
level, type, source, timestamp in attributes.
"""


class MemoryItem(BaseModel):
    id: int = Field(default_factory=lambda: uuid4().int, description="The unique identifier for the memory item")
    content: str = Field(description="Actual content of this memory attribute")
    level: Literal["LTM", "STM"] = Field(description="To which level of granularity this memory attribute belongs to")
    category: str = Field(description="category of this memory attribute")
    type: str = Field(description="attribute name")
    source: str = Field(description="where this memory comes from")
    timestamp: datetime = Field(description="when this memory is created")
    metadata: dict = Field(default_factory=lambda data: {
        "category": data['category'],
        "level": data['level'],
        "type": data['type'],
        "source": data['source'],
        "timestamp": data['timestamp']
    }, description="Additional metadata about the memory item")
    
    def convert_to_sentence(cls, categories: List[str]):
        if cls.category in categories:
            return f"[CATEGORY: {cls.category}] The user's {cls.type} is {cls.content}"
        else:
            return f"The user's {cls.type} is {cls.content}"
    
    def convert_to_attributes(cls):
        pass
        
    def __str__(self):
        return f"MemoryItem(id={self.id}, content={self.content}, level={self.level}, type={self.type}, source={self.source}, timestamp={self.timestamp})"

    def __repr__(self):
        return self.__str__()


class Memory(BaseModel):
    id: int
    user_profile: List[MemoryItem]
    created_at: datetime
    updated_at: datetime

    def __init__(self, id: int, user_profile: List[MemoryItem], created_at: datetime, updated_at: datetime):
        self.id = id
        self.user_profile = user_profile
        self.created_at = created_at
        self.updated_at = updated_at

    def __str__(self):
        return f"Memory(id={self.id}, content={self.content}, created_at={self.created_at}, updated_at={self.updated_at})"

    def __repr__(self):
        return self.__str__()
