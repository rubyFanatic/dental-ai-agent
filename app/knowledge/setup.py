"""
Pinecone Index Setup.

Creates and configures the Pinecone index for the dental agent.
Run this once during initial setup.
"""
import logging
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def setup_index():
    """Create the Pinecone index if it doesn't exist."""
    settings = get_settings()

    if not settings.pinecone_api_key:
        logger.info("Pinecone not configured. Agent will use config-based fallback for service lookups.")
        return

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info(f"Index '{index_name}' already exists.")
        return

    logger.info(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info("Index created successfully.")
