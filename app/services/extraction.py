"""
Phase 2 — Extraction Layer (DUMB)

This layer does ONLY three things:
  1. Extract atomic, factual claims from article text via LLM
  2. Generate embeddings for each claim
  3. Store claims + evidence into the database

It does NOT:
  - Canonicalize
  - Cluster
  - Merge
  - Generate events

All intelligence happens in clustering.py AFTER all claims are stored.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ── Embedding Model (singleton) ──────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def embed_text(text: str) -> List[float]:
    """Returns a normalized 384-dim embedding."""
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()

def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── LLM Helper ───────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("No HF_TOKEN found.")
        return ""
    client = InferenceClient(model=MODEL_ID, token=token)
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

# ── Claim Extraction ─────────────────────────────────────────────

def extract_claims(article_text: str) -> List[Dict[str, Any]]:
    """
    Extracts atomic, verifiable claims from article text.
    Returns list of {text, confidence, evidence_sentence}.
    """
    system_prompt = (
        "You are a strict factual extraction engine.\n"
        "RULES:\n"
        "1. Extract ONLY factual, verifiable assertions.\n"
        "2. DO NOT extract opinions, sentiment, speculation, or editorial framing.\n"
        "3. Claims must be self-contained and atomic. Replace all pronouns with named entities.\n"
        "4. Assign a confidence score (0.0–1.0).\n"
        "5. Provide the exact 'evidence_sentence' — the verbatim sentence from the article that supports this claim.\n"
        "6. Return ONLY valid JSON.\n"
        "Format:\n"
        '{"claims":[{"text":"...","confidence":0.95,"evidence_sentence":"..."}]}'
    )
    user_prompt = f"Extract claims from this article (max 5000 chars):\n\n{article_text[:5000]}"

    raw = call_llm(system_prompt, user_prompt)
    if raw:
        try:
            return json.loads(raw).get("claims", [])
        except json.JSONDecodeError:
            logger.exception("Claim parse failed")
    return []

# ── Relevance Filter ─────────────────────────────────────────────

def claim_relevance(query: str, claim_embedding: List[float]) -> float:
    """Cosine similarity between query embedding and claim embedding."""
    if not query:
        return 1.0
    query_embedding = embed_text(query)
    return cosine_similarity(query_embedding, claim_embedding)

# ── Evidence Deduplication ────────────────────────────────────────

def evidence_hash(claim_id: str, article_id: str, sentence: str) -> str:
    """Deterministic hash for evidence deduplication."""
    raw = f"{claim_id}|{article_id}|{sentence.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ── Main Pipeline: Extract + Store ────────────────────────────────

async def process_and_store_claims(
    prisma,
    article_id: str,
    article_text: str,
    source: str,
    url: str,
    published_at,
    query: str = "",
):
    """
    DUMB extraction pipeline:
      1. Extract claims from article text
      2. For each claim, compute embedding
      3. Filter by relevance to search query
      4. Store claim + evidence (with deduplication)

    NO canonicalization. NO clustering. Those happen later.
    """
    claims = extract_claims(article_text)
    if not claims:
        return []

    inserted = []

    for claim in claims:
        text = claim.get("text", "").strip()
        evidence_sentence = claim.get("evidence_sentence", "").strip()
        confidence = float(claim.get("confidence", 0.8))

        if len(text) < 15:
            continue

        # Generate embedding
        embedding = embed_text(text)

        # Relevance filter (claim-level, not article-level)
        relevance = claim_relevance(query, embedding)
        if query and relevance < 0.45:
            logger.debug(f"Claim rejected (relevance={relevance:.2f}): {text[:60]}")
            continue

        vector_string = "[" + ",".join(map(str, embedding)) + "]"

        # Insert claim (always new — canonicalization happens in clustering)
        created = await prisma.query_raw(
            """
            INSERT INTO "claim"
            ("id","canonicalClaim","confidence","embedding","createdAt")
            VALUES
            (gen_random_uuid()::text, $1, $2, $3::vector, NOW())
            RETURNING id
            """,
            text,
            confidence,
            vector_string,
        )

        if not created:
            continue

        claim_id = created[0]["id"]

        # Evidence deduplication
        ev_hash = evidence_hash(claim_id, article_id, evidence_sentence or text)
        existing = await prisma.query_raw(
            """
            SELECT id FROM "evidence"
            WHERE "claimId" = $1 AND "articleId" = $2
            LIMIT 1
            """,
            claim_id,
            article_id,
        )

        if not existing:
            await prisma.evidence.create(
                data={
                    "claimId": claim_id,
                    "articleId": article_id,
                    "sentence": evidence_sentence or text,
                    "source": source,
                    "url": url,
                    "publishedAt": published_at or datetime.now(),
                    "stance": "MENTION",
                }
            )

        inserted.append(claim_id)

    logger.info(f"Extracted {len(inserted)} claims from article {article_id}")
    return inserted
