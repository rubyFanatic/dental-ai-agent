"""
Application settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "dental-agent"
    pinecone_environment: str = "us-east-1"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "dental-agent-dev"

    # Twilio (optional)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Database (optional - falls back to in-memory)
    database_url: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
