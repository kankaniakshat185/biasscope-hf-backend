"""
Phase 2 Final Quality Gate — Extraction Layer

Pipeline:
  1. Extract claims WITH claim_type classification (single cached LLM call)
  2. Reject OPINION, ANALYSIS, BIOGRAPHICAL, PREDICTION, QUOTE types
  3. Score claim quality (heuristic — no LLM)
  4. Score entity salience (heuristic — no LLM)
  5. Within-article cosine deduplication (sim > 0.92 → discard)
  6. Embedding relevance filter
  7. Store claims + evidence (deduplicated)

All LLM calls go through llm_client.py for caching + analytics.
"""

import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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

def embed_texts_batch(texts: List[str]) -> np.ndarray:
    """Batch embed for efficiency during deduplication."""
    return get_embedding_model().encode(texts, normalize_embeddings=True)

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
        "You are a news intelligence extractor. Extract ONLY verifiable factual claims from the article. "
        "Classify each claim as one of: EVENT, NUMERIC, BIOGRAPHICAL, OPINION, ANALYSIS, PREDICTION, QUOTE.\n\n"
        "STRICT RULES:\n"
        "1. Can this be proven true or false? If not, REJECT IT.\n"
        "2. EVENT: A concrete action that happened (filing, launch, deal, announcement, arrest, explosion)\n"
        "3. NUMERIC: A specific measurable fact ($75B, 42%, 18,712 bitcoin)\n"
        "4. BIOGRAPHICAL: Personal history, marriages, filmography, awards — REJECT THESE\n"
        "5. OPINION: Subjective judgment ('is bad at', 'damaged democracy', 'orchestrated for maximum advantage') — REJECT THESE\n"
        "6. ANALYSIS: Market commentary, predictions about reactions — REJECT THESE\n"
        "7. QUOTE: Direct speech from a person — STORE SEPARATELY\n"
        "8. PREDICTION: Future speculation — STORE SEPARATELY\n\n"
        "Replace ALL pronouns with named entities.\n"
        "Each claim must be a SINGLE atomic fact, not a compound sentence.\n"
        "Return ONLY valid JSON:\n"
        '{"claims":[{"text":"...","claim_type":"EVENT","confidence":0.9,"evidence_sentence":"..."}]}'
    )
    user_prompt = f"Extract claims:\n\n{article_text[:4000]}"

    raw = await cached_llm_call(prisma, "extraction", system_prompt, user_prompt, max_tokens=2048)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            return parsed.get("claims", [])
        except json.JSONDecodeError:
            repaired = repair_truncated_json(raw)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, list):
                        return parsed
                    return parsed.get("claims", [])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Claim parse failed after repair. Raw length: {len(raw)}")
    return []

# ── Claim Quality Gate (heuristic, NO LLM) ───────────────────────

# Words that signal opinion/biographical content even if LLM missed it
OPINION_SIGNALS = {
    'is bad at', 'is good at', 'damaged', 'destroyed', 'ruined',
    'specialises in', 'is known for', 'arguably', 'arguably the',
    'controversial', 'divisive', 'problematic', 'atavistic', 'fearful', 'dumber',
}

BIOGRAPHICAL_SIGNALS = {
    'married', 'divorced', 'born in', 'grew up', 'attended school',
    'starred in', 'appeared in', 'played the role', 'won the award',
    'rising star', 'filmography', 'debut novel', 'dating', 'ex-wife',
    'ex-husband', 'children', 'modelling', 'modeling contracts',
}

def compute_quality_score(text: str) -> float:
    """Score claim quality. Penalize opinion/biographical signals."""
    score = 0.0
    text_lower = text.lower()

    # Issue 9: Reject questions and rhetorical statements
    if '?' in text:
        return 0.0  # Hard reject

    # Penalty: opinion/biographical language that the LLM type gate might miss
    for signal in OPINION_SIGNALS:
        if signal in text_lower:
            return 0.0  # Hard reject
    for signal in BIOGRAPHICAL_SIGNALS:
        if signal in text_lower:
            return 0.10  # Near-certain reject (below 0.40 threshold)

    # Named entities (capitalized multi-word phrases)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
    if len(capitalized) >= 2:
        score += 0.30
    elif len(capitalized) >= 1:
        score += 0.15

    # Numeric data (strong intelligence signal)
    has_numbers = bool(re.search(r'\$[\d,.]+|\d+%|\d{1,3}(?:,\d{3})+|\d+\.\d+', text))
    if has_numbers:
        score += 0.25
    elif bool(re.search(r'\d+', text)):
        score += 0.15

    # Action verbs (event signal)
    action_verbs = [
        'filed', 'signed', 'announced', 'launched', 'acquired', 'sued',
        'appointed', 'resigned', 'merged', 'invested', 'raised', 'sold',
        'denied', 'confirmed', 'reported', 'released', 'purchased', 'agreed',
        'approved', 'rejected', 'blocked', 'expanded', 'partnered', 'settled',
        'exploded', 'arrested', 'convicted', 'charged', 'disclosed', 'filed',
        'began trading', 'went public', 'leased', 'estimated', 'valued',
    ]
    if any(v in text_lower for v in action_verbs):
        score += 0.30

    # Length appropriateness
    word_count = len(text.split())
    if 8 <= word_count <= 35:
        score += 0.15
    elif word_count < 8:
        score += 0.05
    else:
        score += 0.08

    return min(score, 1.0)

# ── Entity Salience Filter (heuristic, NO LLM) ──────────────────

def compute_entity_salience(
    query: str,
    claim_text: str,
    article_title: str = "",
) -> float:
    """
    Score how salient the query entity is to this claim.
    Uses: headline presence (0.4), lead position (0.2),
    mention frequency (0.2), article title match (0.2).
    """
    if not query:
        return 1.0
    query_lower = query.lower()
    claim_lower = claim_text.lower()
    title_lower = article_title.lower() if article_title else ""

    # Check if query entity parts are mentioned (handles "Elon Musk" → "Musk")
    query_parts = query_lower.split()
    any_part_present = any(part in claim_lower for part in query_parts if len(part) > 2)
    full_match = query_lower in claim_lower

    # If neither the full query nor any part is in the claim,
    # give a low base score — let embedding relevance (Gate 4) decide
    if not any_part_present:
        return 0.25  # soft penalty, not hard reject

    score = 0.0

    # Headline/title presence (0.4 weight)
    if title_lower and query_lower in title_lower:
        score += 0.40
    elif title_lower and any(p in title_lower for p in query_parts if len(p) > 2):
        score += 0.25
    else:
        score += 0.10

    if full_match:
        # Position in claim — earlier = more salient (0.2 weight)
        pos = claim_lower.find(query_lower)
        relative_pos = pos / max(len(claim_lower), 1)
        if relative_pos < 0.25:
            score += 0.20
        elif relative_pos < 0.50:
            score += 0.12
        else:
            score += 0.05

        # Mention frequency (0.2 weight)
        freq = claim_lower.count(query_lower)
        if freq >= 2:
            score += 0.20
        elif freq == 1:
            score += 0.10
    else:
        # Partial match (e.g., "Musk" in claim for "Elon Musk" query)
        score += 0.10  # Base for partial match

    # First 5 words prominence (0.2 weight)
    first_words = " ".join(claim_lower.split()[:5])
    if any(p in first_words for p in query_parts if len(p) > 2):
        score += 0.20
    else:
        score += 0.05

    return min(score, 1.0)

# ── Within-Article Deduplication ─────────────────────────────────

DEDUP_THRESHOLD = 0.92

def deduplicate_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate claims from the same article.
    Uses cosine similarity on claim text embeddings.
    If two claims have similarity > DEDUP_THRESHOLD, keep the longer one.
    """
    if len(claims) <= 1:
        return claims

    texts = [c.get("text", "") for c in claims]
    embeddings = embed_texts_batch(texts)

    # Build similarity matrix
    keep = [True] * len(claims)
    for i in range(len(claims)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(claims)):
            if not keep[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > DEDUP_THRESHOLD:
                # Keep the longer, richer claim
                if len(texts[i]) >= len(texts[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    deduped = [c for c, k in zip(claims, keep) if k]
    if len(deduped) < len(claims):
        logger.info(f"[DEDUP] {len(claims)} → {len(deduped)} claims (removed {len(claims) - len(deduped)} duplicates)")
    return deduped

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
    article_title: str = "",
):
    """
    Final quality gate extraction pipeline with cached LLM calls.
    """
    claims = await extract_claims(prisma, article_text)
    if not claims:
        return []

    # ── Gate 0: Within-Article Deduplication ──
    claims = deduplicate_claims(claims)

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

        # ── Gate 2: Quality Score (with opinion/biographical penalty) ──
        quality = compute_quality_score(text)
        if quality < 0.40:
            stats["quality_rejected"] += 1
            continue

        # ── Gate 3: Entity Salience ──
        if query:
            salience = compute_entity_salience(query, text, article_title)
            if salience < 0.40:
                stats["salience_rejected"] += 1
                continue

        # ── Gate 4: Embedding Relevance ──
        embedding = embed_text(text)
        relevance = claim_relevance(query, embedding)
        if query and relevance < 0.40:
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
        f"Extraction [{source}]: {stats['stored']}/{stats['total']} stored | "
        f"type={stats['type_rejected']} quality={stats['quality_rejected']} "
        f"salience={stats['salience_rejected']} relevance={stats['relevance_rejected']}"
    )
    return inserted
