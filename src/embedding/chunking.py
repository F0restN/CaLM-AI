import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import logger

class Chunking:
    def __init__(self, text: str, chunker_type: int = 0, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Chunking class
        
        Args:
            text (str): Text to be chunked
            chunker_type (int): Type of chunker (0: Recursive, 1: Character, 2: Markdown)
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        self.text = text
        self.chunker_type = chunker_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self) -> List[str]:
        """
        Chunk the text into smaller chunks
        """
        text_splitter = None
        match self.chunker_type:
            case 0:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            case 1:
                text_splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            case 2:
                text_splitter = MarkdownTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
        if text_splitter is None:
            error_msg = f"Invalid chunker type: {self.chunker_type}"
            logger.error(error_msg)
        
        return text_splitter.split_text(self.text)

