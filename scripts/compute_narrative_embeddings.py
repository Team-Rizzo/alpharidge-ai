#!/usr/bin/env python3
"""
Compute narrative embeddings using all-MiniLM-L6-v2 and store in the database.

Uses the same model as the miner so embeddings are in the same vector space.
Run at deploy time and daily via cron:
    python scripts/compute_narrative_embeddings.py

Requires:
    - sentence-transformers
    - Database connection via DATABASE_URL env var (or .env file)
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


async def main():
    from sentence_transformers import SentenceTransformer
    from prisma import Prisma

    print(f"Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    db = Prisma()
    await db.connect()

    try:
        narratives = await db.narrative.find_many()
        print(f"Found {len(narratives)} narratives in database")

        updated = 0
        skipped = 0
        for n in narratives:
            if n.embedding and len(n.embedding) == EMBEDDING_DIM and n.embeddingModel == EMBEDDING_MODEL:
                skipped += 1
                continue

            keywords = n.keywords if isinstance(n.keywords, list) else []
            text = f"{n.name}. {n.description or ''}. {', '.join(keywords)}"
            embedding = model.encode(text, normalize_embeddings=True).tolist()

            await db.narrative.update(
                where={"id": n.id},
                data={"embedding": embedding, "embeddingModel": EMBEDDING_MODEL},
            )
            updated += 1
            print(f"  [{n.id}] {n.name} — embedded ({len(embedding)}d)")

        print(f"\nDone: {updated} updated, {skipped} already had embeddings")

        # Also compute for candidates that lack embeddings
        candidates = await db.narrativecandidate.find_many(
            where={"promoted": False},
        )
        cand_updated = 0
        for c in candidates:
            if c.embedding and len(c.embedding) == EMBEDDING_DIM:
                continue
            embedding = model.encode(c.keyword, normalize_embeddings=True).tolist()
            await db.narrativecandidate.update(
                where={"id": c.id},
                data={"embedding": embedding},
            )
            cand_updated += 1

        if cand_updated:
            print(f"Updated {cand_updated} candidate embeddings")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
