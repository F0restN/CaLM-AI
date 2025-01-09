import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
import torch
from utils.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel

class EmbeddingModels:
    
    def __init__(self):
        pass
    
    def get_bge_embedding(
        self,
        model_name: str = "BAAI/bge-m3",
        encode_kwargs: Dict = {"normalize_embeddings": True},
    ) -> HuggingFaceEmbeddings:
        """Get BGE Embedding Instance via HF Repository"""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_built() else "cpu")
        )
        
        logger.info(f"Using device: {device}")
        logger.info(f"Loading BGE embedding model: {model_name}")

        model_kwargs = {
            "device": device,
        }

        try:
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            logger.info("Successfully loaded BGE embedding model")
            return embedding
        except Exception as e:
            logger.error(f"Failed to load BGE embedding model: {str(e)}")
            raise

    def get_jina_embedding(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        encode_kwargs: Dict = {"normalize_embeddings": True},
    ) -> HuggingFaceEmbeddings:
        """Get Jina Embedding Instance via HF Repository"""
    
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_built() else "cpu")
        )
        
        logger.info(f"Using device: {device}")
        logger.info(f"Loading Jina embedding model: {model_name}")

        encode_kwargs["device"] = device
        encode_kwargs["trust_remote_code"] = True
        
        try:
            model = AutoModel.from_pretrained("jinaai/xlm-roberta-flash-implementation")
            embedding = AutoModel.from_pretrained(model_name, encode_kwargs)
            logger.info("Successfully loaded Jina embedding model")
            return embedding
        except Exception as e:
            logger.error(f"Failed to load Jina embedding model: {str(e)}")
            raise

    def main(self):
        '''
        Test the embedding model
        
        @Evaluation:
        - Previous similarity search results shows sometimes it returns nothing e.g. "what is ADRD" (Should be able to return something)
        
        @TODO:
        - Evaluate the retrieval results
        - Upgrade the embedding model (jina-embeddings-v3)
        '''


if __name__ == "__main__":
    embeddingModels = EmbeddingModels()
    embeddingModels.get_bge_embedding()

