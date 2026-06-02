"""
Phase 2 Final — Extraction Layer (Hardened)

Pipeline:
  1. Extract claims WITH claim_type classification (single cached LLM call)
  2. Reject OPINION, ANALYSIS, BIOGRAPHICAL, PREDICTION types
  3. Score claim quality (heuristic — no LLM)
  4. Score entity salience (heuristic — no LLM)
  5. Filter low-quality and low-salience claims
  6. Generate embeddings + cosine relevance filter
  7. Store claims + evidence (deduplicated)

All LLM calls go through llm_client.py for caching + analytics.
"""

import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.llm_client import cached_llm_call

logger = logging.getLogger(__name__)

# Types that are allowed into the intelligence pipeline
ALLOWED_CLAIM_TYPES = {"EVENT", "NUMERIC"}

# ── Embedding Model (singleton) ──────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def embed_text(text: str) -> List[float]:
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()

def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── JSON Repair ──────────────────────────────────────────────────

def repair_truncated_json(raw: str) -> str:
    raw = raw.strip()
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    last_complete = raw.rfind('},')
    if last_complete == -1:
        last_complete = raw.rfind('}')
    if last_complete > 0:
        truncated = raw[:last_complete + 1]
        if not truncated.endswith(']}'):
            truncated += ']}'
        try:
            json.loads(truncated)
            return truncated
        except json.JSONDecodeError:
            pass
    return ""

# ── Claim Extraction (cached) ────────────────────────────────────

async def extract_claims(prisma, article_text: str) -> List[Dict[str, Any]]:
    """
    Single CACHED LLM call that extracts claims AND classifies them.
    Returns: [{text, claim_type, confidence, evidence_sentence}]
    """
    system_prompt = (
        "Extract factual claims from the text. Classify each as: "
        "EVENT, NUMERIC, BIOGRAPHICAL, OPINION, ANALYSIS, PREDICTION, or QUOTE. "
        "Replace pronouns with named entities. "
        "Return ONLY valid JSON:\n"
        '{"claims":[{"text":"...","claim_type":"EVENT","confidence":0.9,"evidence_sentence":"..."}]}'
    )
    user_prompt = f"Extract claims:\n\n{article_text[:4000]}"

    raw = await cached_llm_call(prisma, "extraction", system_prompt, user_prompt, max_tokens=2048)
    if raw:
        try:
            return json.loads(raw).get("claims", [])
        except json.JSONDecodeError:
            repaired = repair_truncated_json(raw)
            if repaired:
                try:
                    return json.loads(repaired).get("claims", [])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Claim parse failed after repair. Raw length: {len(raw)}")
    return []

# ── Claim Quality Gate (heuristic, NO LLM) ───────────────────────

def compute_quality_score(text: str) -> float:
    score = 0.0
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
    if len(capitalized) >= 2:
        score += 0.30
    elif len(capitalized) >= 1:
        score += 0.15
    has_numbers = bool(re.search(r'\d+', text))
    if has_numbers:
        score += 0.20
    action_verbs = ['filed', 'signed', 'announced', 'launched', 'acquired', 'sued',
                    'appointed', 'resigned', 'merged', 'invested', 'raised', 'sold',
                    'denied', 'confirmed', 'reported', 'released', 'purchased', 'agreed',
                    'approved', 'rejected', 'blocked', 'expanded', 'partnered', 'settled']
    text_lower = text.lower()
    if any(v in text_lower for v in action_verbs):
        score += 0.30
    word_count = len(text.split())
    if 8 <= word_count <= 30:
        score += 0.20
    elif word_count < 8:
        score += 0.05
    else:
        score += 0.10
    return min(score, 1.0)

# ── Entity Salience Filter (heuristic, NO LLM) ──────────────────

def compute_entity_salience(query: str, claim_text: str) -> float:
    if not query:
        return 1.0
    query_lower = query.lower()
    claim_lower = claim_text.lower()
    words = claim_lower.split()
    score = 0.0
    if query_lower not in claim_lower:
        return 0.0
    pos = claim_lower.find(query_lower)
    relative_pos = pos / max(len(claim_lower), 1)
    if relative_pos < 0.3:
        score += 0.40
    elif relative_pos < 0.6:
        score += 0.25
    else:
        score += 0.10
    freq = claim_lower.count(query_lower)
    if freq >= 2:
        score += 0.20
    elif freq == 1:
        score += 0.10
    first_words = " ".join(words[:4])
    if query_lower in first_words:
        score += 0.30
    return min(score, 1.0)

# ── Relevance Filter ─────────────────────────────────────────────

def claim_relevance(query: str, claim_embedding: List[float]) -> float:
    if not query:
        return 1.0
    query_embedding = embed_text(query)
    return cosine_similarity(query_embedding, claim_embedding)

# ── Main Pipeline ─────────────────────────────────────────────────

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
    Hardened extraction pipeline with cached LLM calls.
    """
    claims = await extract_claims(prisma, article_text)
    if not claims:
        return []

    inserted = []
    stats = {"total": len(claims), "type_rejected": 0, "quality_rejected": 0,
             "salience_rejected": 0, "relevance_rejected": 0, "stored": 0}

    for claim in claims:
        text = claim.get("text", "").strip()
        claim_type = claim.get("claim_type", "EVENT").upper()
        evidence_sentence = claim.get("evidence_sentence", "").strip()
        confidence = float(claim.get("confidence", 0.8))

        if len(text) < 15:
            continue

        # ── Gate 1: Claim Type Filter ──
        if claim_type not in ALLOWED_CLAIM_TYPES:
            stats["type_rejected"] += 1
            continue

        # ── Gate 2: Quality Score ──
        quality = compute_quality_score(text)
        if quality < 0.60:
            stats["quality_rejected"] += 1
            continue

        # ── Gate 3: Entity Salience ──
        if query:
            salience = compute_entity_salience(query, text)
            if salience < 0.50:
                stats["salience_rejected"] += 1
                continue

        # ── Gate 4: Embedding Relevance ──
        embedding = embed_text(text)
        relevance = claim_relevance(query, embedding)
        if query and relevance < 0.45:
            stats["relevance_rejected"] += 1
            continue

        vector_string = "[" + ",".join(map(str, embedding)) + "]"

        created = await prisma.query_raw(
            """
            INSERT INTO "claim"
            ("id","canonicalClaim","claimType","qualityScore","confidence","embedding","createdAt")
            VALUES
            (gen_random_uuid()::text, $1, $2, $3, $4, $5::vector, NOW())
            RETURNING id
            """,
            text,
            claim_type,
            quality,
            confidence,
            vector_string,
        )

        if not created:
            continue

        claim_id = created[0]["id"]

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
        stats["stored"] += 1

    logger.info(
        f"Extraction [{article_id}]: {stats['stored']}/{stats['total']} stored | "
        f"type={stats['type_rejected']} quality={stats['quality_rejected']} "
        f"salience={stats['salience_rejected']} relevance={stats['relevance_rejected']}"
    )
    return inserted
