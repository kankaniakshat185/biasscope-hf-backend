"""
Phase 2.5 — Clustering & Intelligence Layer

Pipeline:
  1. Load claims + embeddings
  2. Normalize → HDBSCAN clustering
  3. LLM cluster merge pass (1 call total)
  4. LLM canonical claim per cluster (1 call per cluster)
  5. TF-IDF + NER event title generation (0 LLM calls)
  6. Event eligibility check
  7. Cross-source consensus scoring
  8. Event importance ranking
  9. Store everything

LLM Budget:
  - 1 merge call
  - N canonicalization calls (N = number of clusters, typically 5-10)
  - 0 event naming calls (TF-IDF replaces LLM)
  - 0 cluster title calls (uses canonical claim)
  Total: ~6-11 calls (down from ~45+)
"""

import os
import re
import json
import logging
import numpy as np
from collections import Counter
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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

def parse_json_safe(raw: str, fallback: dict = None) -> dict:
    if not raw:
        return fallback or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback or {}

# ── Section 7: TF-IDF Event Title Generation (NO LLM) ────────────

def generate_event_title_tfidf(claim_texts: List[str]) -> str:
    """
    Generate event title from top TF-IDF keywords + NER entities.
    Zero LLM cost.
    """
    if not claim_texts:
        return "Unclassified Event"

    # Extract named entities (capitalized multi-word phrases)
    all_text = " ".join(claim_texts)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', all_text)
    entity_counts = Counter(entities)

    # Remove generic words
    stopwords = {"The", "This", "That", "These", "Those", "According", "However",
                 "While", "After", "Before", "During", "Between", "About", "Also"}
    for sw in stopwords:
        entity_counts.pop(sw, None)

    # Top entities
    top_entities = [e for e, _ in entity_counts.most_common(3)]

    # TF-IDF for action keywords
    try:
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(claim_texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
    except Exception:
        top_keywords = []

    # Action verbs for context
    action_map = {
        "filed": "Filing", "ipo": "IPO", "merger": "Merger", "acquisition": "Acquisition",
        "lawsuit": "Lawsuit", "sued": "Lawsuit", "launch": "Launch", "launched": "Launch",
        "agreement": "Agreement", "deal": "Deal", "partnership": "Partnership",
        "compute": "Compute", "signed": "Agreement", "controversy": "Controversy",
        "political": "Political", "election": "Election", "resign": "Resignation",
        "appointed": "Appointment", "invest": "Investment", "raised": "Funding",
        "acquired": "Acquisition", "bankruptcy": "Bankruptcy", "fraud": "Fraud",
    }

    action_word = ""
    for kw in top_keywords:
        for trigger, label in action_map.items():
            if trigger in kw.lower():
                action_word = label
                break
        if action_word:
            break

    # Build title
    if top_entities and action_word:
        if len(top_entities) >= 2:
            title = f"{top_entities[0]}–{top_entities[1]} {action_word}"
        else:
            title = f"{top_entities[0]} {action_word}"
    elif top_entities:
        title = " ".join(top_entities[:3])
    else:
        # Fallback: first claim truncated
        title = claim_texts[0][:60]

    return title.strip()

# ══════════════════════════════════════════════════════════════════
# STEP 1: CLAIM CLUSTERING
# ══════════════════════════════════════════════════════════════════

async def run_claim_clustering(prisma):
    """
    1. Load claims
    2. HDBSCAN on normalized embeddings
    3. LLM merge pass (1 call)
    4. Canonical claim generation per cluster (1 call each)
    5. Store clusters with canonical claims + consensus scores
    """
    logger.info("=== Phase 2.5 Clustering Pipeline ===")

    claims = await prisma.query_raw("""
        SELECT id, "canonicalClaim", embedding::text
        FROM "claim"
        WHERE "clusterId" IS NULL
    """)

    if not claims or len(claims) < 3:
        logger.info(f"Only {len(claims) if claims else 0} unclustered claims — skipping.")
        return

    ids, texts, vectors = [], [], []
    for c in claims:
        try:
            vec = [float(x) for x in c["embedding"][1:-1].split(",")]
            vectors.append(vec)
            ids.append(c["id"])
            texts.append(c["canonicalClaim"])
        except (ValueError, IndexError):
            continue

    if len(vectors) < 3:
        return

    X = np.array(vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    # HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X)

    groups: Dict[int, List[Dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(label, [])
        groups[label].append({"id": ids[idx], "text": texts[idx]})

    if not groups:
        logger.info("No clusters formed.")
        return

    noise_count = sum(1 for l in labels if l == -1)
    logger.info(f"HDBSCAN: {len(groups)} clusters, {noise_count} noise claims from {len(ids)} total.")

    # LLM Merge Pass (1 call)
    groups = _llm_merge_pass(groups)
    logger.info(f"After merge: {len(groups)} clusters.")

    # Per-cluster: canonicalize + store
    for label, members in groups.items():
        claim_texts = [m["text"] for m in members]

        # Canonical claim (1 LLM call per cluster)
        canonical = _generate_canonical_claim(claim_texts)

        # Cluster title = TF-IDF (0 LLM calls)
        title = generate_event_title_tfidf(claim_texts)

        # Create cluster
        cluster_record = await prisma.claimcluster.create(
            data={
                "title": title,
                "canonicalClaim": canonical,
            }
        )

        # Assign claims to cluster
        for member in members:
            await prisma.claim.update(
                where={"id": member["id"]},
                data={
                    "clusterId": cluster_record.id,
                    "canonicalClaim": canonical,
                },
            )

    logger.info("=== Clustering Complete ===")

# ── LLM Merge Pass (1 total call) ────────────────────────────────

def _llm_merge_pass(groups: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    if len(groups) <= 1:
        return groups

    payload = []
    for lbl in list(groups.keys()):
        payload.append({
            "cluster_id": int(lbl),
            "claims": [m["text"] for m in groups[lbl]][:8],
        })

    system_prompt = (
        "Merge clusters that describe the SAME real-world event. "
        "Return JSON: "
        '{"merge_groups": [[id1, id2], [id3, id4]]} '
        "If no merges needed: "
        '{"merge_groups": []}'
    )

    raw = call_llm(system_prompt, f"Clusters:\n{json.dumps(payload)}")
    data = parse_json_safe(raw, {"merge_groups": []})

    for group in data.get("merge_groups", []):
        if not group or len(group) < 2:
            continue
        target = group[0]
        for src in group[1:]:
            if src in groups and target in groups:
                groups[target].extend(groups[src])
                del groups[src]

    return groups

# ── Section 8: Canonical Claim (per-cluster, NOT per-claim) ──────

def _generate_canonical_claim(claim_texts: List[str]) -> str:
    if len(claim_texts) == 1:
        return claim_texts[0]

    system_prompt = (
        "Canonicalize these related claims into ONE definitive, factual claim. "
        "Return JSON: "
        '{"canonical_claim": "..."}'
    )
    raw = call_llm(system_prompt, json.dumps(claim_texts[:15]), max_tokens=256)
    data = parse_json_safe(raw)
    return data.get("canonical_claim", claim_texts[0])

# ══════════════════════════════════════════════════════════════════
# STEP 2: EVENT DETECTION
# ══════════════════════════════════════════════════════════════════

async def run_event_detection(prisma):
    """
    1. Load clusters with claims + evidence
    2. Eligibility check
    3. Compute cross-source consensus
    4. Generate event title (TF-IDF, 0 LLM)
    5. Generate event summary (1 LLM call per event)
    6. Importance ranking
    7. Store
    """
    logger.info("=== Phase 2.5 Event Detection ===")

    clusters = await prisma.claimcluster.find_many(
        where={"eventId": None},
        include={
            "claims": {
                "include": {"evidence": True}
            }
        },
    )

    if not clusters:
        return

    events_created = 0

    for cluster in clusters:
        if not cluster.claims:
            continue

        # Stats
        claim_count = len(cluster.claims)
        all_evidence = []
        for c in cluster.claims:
            all_evidence.extend(c.evidence)

        evidence_count = len(all_evidence)
        sources = set(e.source for e in all_evidence)
        source_count = len(sources)

        # ── Event Eligibility ──
        if source_count < 2 or claim_count < 2 or evidence_count < 3:
            continue

        # ── Section 9: Cross-Source Consensus ──
        consensus_score = source_count / max(evidence_count, 1)
        consensus_score = min(consensus_score, 1.0)

        # Update cluster consensus
        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"consensusScore": consensus_score},
        )

        # ── Event Title (TF-IDF, 0 LLM) ──
        claim_texts = [c.canonicalClaim for c in cluster.claims]
        event_title = generate_event_title_tfidf(claim_texts)

        # ── Event Summary (1 LLM call) ──
        evidence_sentences = list(set(e.sentence for e in all_evidence))[:8]
        summary_prompt = json.dumps({
            "title": event_title,
            "canonical_claim": getattr(cluster, 'canonicalClaim', '') or cluster.title,
            "sources": list(sources),
        })
        raw_summary = call_llm(
            "Write a 1-sentence news summary for this event. Return JSON: "
            '{"summary": "..."}',
            summary_prompt,
            max_tokens=128,
        )
        event_summary = parse_json_safe(raw_summary).get("summary", "")

        # ── Importance Score ──
        publisher_diversity = source_count
        importance = (
            source_count * 0.30
            + publisher_diversity * 0.20
            + evidence_count * 0.15
            + claim_count * 0.15
            + consensus_score * 0.20
        )
        if publisher_diversity > 3:
            importance += 2.0

        # ── Store Event ──
        event_record = await prisma.event.create(
            data={
                "title": event_title,
                "description": event_summary,
                "importanceScore": importance,
            }
        )

        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"eventId": event_record.id},
        )

        events_created += 1

    logger.info(f"=== Event Detection Complete: {events_created} events ===")
