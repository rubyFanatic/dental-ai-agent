"""
Seed the Pinecone knowledge base with practice data.

Usage:
    python scripts/seed_knowledge.py
    python scripts/seed_knowledge.py --practice cary-medspa-01

If Pinecone is not configured, prints what WOULD be seeded.
"""
import json
import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def seed_from_file(filepath: str, practice_id: str):
    """Seed Pinecone from a JSON data file."""
    from app.config.settings import get_settings
    settings = get_settings()

    with open(filepath) as f:
        data = json.load(f)

    documents = data["documents"]
    logger.info(f"Loaded {len(documents)} documents for {practice_id}")

    if not settings.pinecone_api_key:
        logger.warning("Pinecone not configured. Printing documents that WOULD be seeded:")
        for doc in documents:
            logger.info(f"  [{doc['id']}] {doc['category']}: {doc['content'][:80]}...")
        logger.info("Set PINECONE_API_KEY in .env to actually seed.")
        return

    # Initialize Pinecone and OpenAI
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI

    pc = Pinecone(api_key=settings.pinecone_api_key)
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # Create index if it doesn't exist
    index_name = settings.pinecone_index_name
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created. Waiting for it to be ready...")
        import time
        time.sleep(10)

    index = pc.Index(index_name)

    # Generate embeddings and upsert
    vectors = []
    for doc in documents:
        logger.info(f"Embedding: {doc['id']}")

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=doc["content"],
        )
        embedding = response.data[0].embedding

        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                "content": doc["content"],
                "service": doc.get("service") or "",
                "category": doc["category"],
                "practice_id": practice_id,
            },
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=practice_id)
        logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    logger.info(f"Done! Seeded {len(vectors)} vectors into namespace '{practice_id}'")

    # Verify
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Seed Pinecone knowledge base")
    parser.add_argument(
        "--practice",
        default="apex-dental-01",
        help="Practice ID to seed (default: apex-dental-01)",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Path to JSON data file (auto-detected from data/ directory if not specified)",
    )
    args = parser.parse_args()

    # Auto-detect data file
    if args.data_file:
        filepath = args.data_file
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        # Map practice IDs to data files
        data_files = {
            "apex-dental-01": "apex_dental_services.json",
        }
        filename = data_files.get(args.practice)
        if not filename:
            logger.error(f"No data file mapped for practice '{args.practice}'")
            logger.info(f"Available: {list(data_files.keys())}")
            sys.exit(1)
        filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        sys.exit(1)

    seed_from_file(filepath, args.practice)


if __name__ == "__main__":
    main()
