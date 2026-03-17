"""
Knowledge Base Retriever — RAG lookup against Pinecone.

Each practice has its own Pinecone namespace containing:
- Service descriptions, durations, preparation instructions
- FAQs
- Insurance details
- Post-procedure care instructions

For the MVP, we provide a fallback that works without Pinecone
using the practice config data directly.
"""
import logging
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def retrieve_service_info(query: str, practice_id: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve relevant service information from the knowledge base.

    Args:
        query: The patient's question or service name
        practice_id: Practice namespace to search in
        top_k: Number of results to return

    Returns:
        List of matching documents with content and metadata
    """
    settings = get_settings()

    if settings.pinecone_api_key:
        return _retrieve_from_pinecone(query, practice_id, top_k)
    else:
        logger.info("Pinecone not configured — using config-based fallback")
        return _retrieve_fallback(query, practice_id)


def _retrieve_from_pinecone(query: str, practice_id: str, top_k: int) -> list[dict]:
    """Query Pinecone with embedding similarity search."""
    from pinecone import Pinecone
    from openai import OpenAI

    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Generate embedding for the query
    openai_client = OpenAI(api_key=settings.openai_api_key)
    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    # Query Pinecone with the practice's namespace
    results = index.query(
        vector=query_embedding,
        namespace=practice_id,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "content": match.metadata.get("content", ""),
            "service": match.metadata.get("service", ""),
            "category": match.metadata.get("category", ""),
            "score": match.score,
        }
        for match in results.matches
    ]


def _retrieve_fallback(query: str, practice_id: str) -> list[dict]:
    """Fallback retriever using practice config data when Pinecone isn't configured."""
    from app.config.practice_config import get_practice

    practice = get_practice(practice_id)
    query_lower = query.lower()

    results = []
    for service in practice["services"]:
        if service.lower() in query_lower or query_lower in service.lower():
            duration = practice["booking_rules"]["slot_duration_minutes"].get(service, 30)
            results.append({
                "content": f"{service}: Duration {duration} minutes.",
                "service": service,
                "category": "service",
                "score": 0.9,
            })

    return results[:3]
