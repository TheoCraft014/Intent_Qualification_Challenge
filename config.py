import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://LLMQuery.services.ai.azure.com/")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-5")

# Vector Embedding Configuration
EMBEDDER_MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_PATH = "companies.jsonl"