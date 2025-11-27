"""
Global configuration for the Earnings Call Interpreter project.
Use environment variables or a .env file to manage secrets.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# Project root directory (earnings_call_interpreter/)
BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /.../earnings_call_interpreter
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

@dataclass
class VectorDBConfig:
    persist_directory: str = str(CHROMA_DIR)
    collection_name: str = "earnings_calls"

@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    api_key: str = os.getenv("GEMINI_API_KEY", "")

@dataclass
class GraphDBConfig:
    uri: str = os.getenv("NEO4J_URI", "")
    user: str = os.getenv("NEO4J_USER", "")
    password: str = os.getenv("NEO4J_PASSWORD", "")

@dataclass
class Settings:
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = field(default_factory=GraphDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


settings = Settings()
