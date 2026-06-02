"""
Phase 2.5 — Extraction Layer (Hardened)

Pipeline:
  1. Extract claims WITH claim_type classification (single LLM call)
  2. Reject OPINION, ANALYSIS, BIOGRAPHICAL, PREDICTION types
  3. Score claim quality (heuristic — no LLM)
  4. Score entity salience (heuristic — no LLM)
  5. Filter low-quality and low-salience claims
  6. Generate embeddings + cosine relevance filter
  7. Store claims + evidence (deduplicated)

This layer is deliberately STRICT.
The goal is to prevent garbage from entering the clustering pipeline.
"""

import os
import re
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

# ── LLM Helper ───────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
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

# ── Section 3: Claim Extraction WITH Type Classification ─────────

def extract_claims(article_text: str) -> List[Dict[str, Any]]:
    """
    Single LLM call that extracts claims AND classifies them.
    Returns: [{text, claim_type, confidence, evidence_sentence}]
    """
    system_prompt = (
        "You are a strict factual extraction engine.\n"
        "RULES:\n"
        "1. Extract assertions from the text.\n"
        "2. Classify each claim into exactly one type:\n"
        "   EVENT — something that happened (e.g., 'SpaceX filed for IPO')\n"
        "   NUMERIC — a specific measurable fact (e.g., 'Musk will retain 42% ownership')\n"
        "   BIOGRAPHICAL — personal history or trivia (e.g., 'Talulah Riley married Elon Musk')\n"
        "   OPINION — subjective judgment (e.g., 'Elon Musk is bad at politics')\n"
        "   ANALYSIS — editorial interpretation (e.g., 'Retail investors love Musk')\n"
        "   PREDICTION — future speculation (e.g., 'Musk could become a trillionaire')\n"
        "   QUOTE — direct speech attribution (e.g., 'Musk said humanity depends on it')\n"
        "3. Claims must be self-contained. Replace all pronouns with named entities.\n"
        "4. Assign a confidence score (0.0–1.0).\n"
        "5. Provide the exact evidence_sentence from the article.\n"
        "6. Return ONLY valid JSON.\n"
        "Format:\n"
        '{"claims":[{"text":"...","claim_type":"EVENT","confidence":0.95,"evidence_sentence":"..."}]}'
    )
    user_prompt = f"Extract and classify claims:\n\n{article_text[:5000]}"

    raw = call_llm(system_prompt, user_prompt)
    if raw:
        try:
            return json.loads(raw).get("claims", [])
        except json.JSONDecodeError:
            logger.exception("Claim parse failed")
    return []

# ── Section 4: Claim Quality Gate (heuristic, NO LLM) ────────────

def compute_quality_score(text: str) -> float:
    """
    Heuristic quality score based on:
      - Verifiability: contains named entities / proper nouns?
      - Specificity: contains numbers, dates, or concrete details?
      - Newsworthiness: action verbs present?
      - Length: too short = vague, too long = editorial
    Returns 0.0–1.0
    """
    score = 0.0

    # Verifiability: capitalized words (proxy for named entities)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
    if len(capitalized) >= 2:
        score += 0.30
    elif len(capitalized) >= 1:
        score += 0.15

    # Specificity: numbers, dates, percentages
    has_numbers = bool(re.search(r'\d+', text))
    if has_numbers:
        score += 0.20

    # Newsworthiness: action verbs
    action_verbs = ['filed', 'signed', 'announced', 'launched', 'acquired', 'sued',
                    'appointed', 'resigned', 'merged', 'invested', 'raised', 'sold',
                    'denied', 'confirmed', 'reported', 'released', 'purchased', 'agreed',
                    'approved', 'rejected', 'blocked', 'expanded', 'partnered', 'settled']
    text_lower = text.lower()
    if any(v in text_lower for v in action_verbs):
        score += 0.30

    # Length penalty
    word_count = len(text.split())
    if 8 <= word_count <= 30:
        score += 0.20
    elif word_count < 8:
        score += 0.05  # too vague
    else:
        score += 0.10  # too editorial

    return min(score, 1.0)

# ── Section 5: Entity Salience Filter (heuristic, NO LLM) ────────

def compute_entity_salience(query: str, claim_text: str) -> float:
    """
    Measures how central the queried entity is to this claim.
    High: "SpaceX filed IPO paperwork" (for query "Elon Musk")
    Low:  "Talulah Riley article mentioning Elon Musk" (peripheral mention)
    Returns 0.0–1.0
    """
    if not query:
        return 1.0

    query_lower = query.lower()
    claim_lower = claim_text.lower()
    words = claim_lower.split()

    score = 0.0

    # Is query entity in the claim at all?
    if query_lower not in claim_lower:
        return 0.0

    # Position: entity appears in first half of claim = more salient
    pos = claim_lower.find(query_lower)
    relative_pos = pos / max(len(claim_lower), 1)
    if relative_pos < 0.3:
        score += 0.40  # subject position
    elif relative_pos < 0.6:
        score += 0.25
    else:
        score += 0.10  # mentioned late = peripheral

    # Frequency
    freq = claim_lower.count(query_lower)
    if freq >= 2:
        score += 0.20
    elif freq == 1:
        score += 0.10

    # Is entity the grammatical subject? (rough heuristic: first 3 words)
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
    Hardened extraction pipeline:
      1. Extract + classify claims (single LLM call)
      2. Type filter — reject OPINION, ANALYSIS, BIOGRAPHICAL, PREDICTION, QUOTE
      3. Quality gate — reject score < 0.60
      4. Entity salience — reject < 0.50
      5. Embedding + cosine relevance — reject < 0.45
      6. Store claim + evidence (deduplicated)
    """
    claims = extract_claims(article_text)
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

        # Store claim
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

        # Evidence deduplication
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
        f"Extraction complete for article {article_id}: "
        f"{stats['stored']}/{stats['total']} stored | "
        f"type_rejected={stats['type_rejected']} quality_rejected={stats['quality_rejected']} "
        f"salience_rejected={stats['salience_rejected']} relevance_rejected={stats['relevance_rejected']}"
    )
    return inserted
