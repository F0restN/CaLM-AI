from typing import Dict
from langchain_ollama import OllamaEmbeddings
from utils.logger import logger


def get_nomic_embedding():
    return OllamaEmbeddings(model="nomic-embed-text:latest")

