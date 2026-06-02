"""
Phase 2 Final — Clustering & Intelligence Layer

Architecture:
  - Claims keep their ORIGINAL text (never overwritten)
  - ClaimCluster owns the canonical claim
  - Evidence is always the original sentence from extraction
  - Event titles via TF-IDF (0 LLM)
  - Canonical claims via LLM (1 per cluster)
  - Aggregation metrics computed at cluster level

Pipeline:
  1. Load unclustered claims + embeddings
  2. Normalize → HDBSCAN (min_cluster_size=2)
  3. LLM merge pass (1 call)
  4. Per cluster: generate canonical claim (LLM), title (TF-IDF)
  5. Store cluster (canonical on CLUSTER, not on claims)
  6. Event eligibility → event creation
  7. Consensus + importance scoring
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

# ── TF-IDF Event Title (0 LLM calls) ─────────────────────────────

def generate_event_title_tfidf(claim_texts: List[str]) -> str:
    if not claim_texts:
        return "Unclassified Event"

    all_text = " ".join(claim_texts)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', all_text)
    entity_counts = Counter(entities)

    stopwords = {"The", "This", "That", "These", "Those", "According", "However",
                 "While", "After", "Before", "During", "Between", "About", "Also"}
    for sw in stopwords:
        entity_counts.pop(sw, None)

    top_entities = [e for e, _ in entity_counts.most_common(3)]

    try:
        vectorizer = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(claim_texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
    except Exception:
        top_keywords = []

    action_map = {
        "filed": "Filing", "ipo": "IPO", "merger": "Merger", "acquisition": "Acquisition",
        "lawsuit": "Lawsuit", "sued": "Lawsuit", "launch": "Launch", "launched": "Launch",
        "agreement": "Agreement", "deal": "Deal", "partnership": "Partnership",
        "compute": "Compute", "signed": "Agreement", "controversy": "Controversy",
        "political": "Political", "election": "Election", "resign": "Resignation",
        "appointed": "Appointment", "invest": "Investment", "raised": "Funding",
        "acquired": "Acquisition", "bankruptcy": "Bankruptcy", "fraud": "Fraud",
        "safety": "Safety", "regulation": "Regulation", "ban": "Ban",
        "explosion": "Explosion", "failure": "Failure", "crash": "Crash",
    }

    action_word = ""
    for kw in top_keywords:
        for trigger, label in action_map.items():
            if trigger in kw.lower():
                action_word = label
                break
        if action_word:
            break

    if top_entities and action_word:
        if len(top_entities) >= 2:
            title = f"{top_entities[0]}–{top_entities[1]} {action_word}"
        else:
            title = f"{top_entities[0]} {action_word}"
    elif top_entities:
        title = " ".join(top_entities[:3])
    else:
        title = claim_texts[0][:60]

    return title.strip()

# ══════════════════════════════════════════════════════════════════
# STEP 1: CLAIM CLUSTERING
# ══════════════════════════════════════════════════════════════════

async def run_claim_clustering(prisma):
    logger.info("=== Phase 2 Final — Clustering Pipeline ===")

    # Load only unclustered claims
    claims = await prisma.query_raw("""
        SELECT id, "canonicalClaim", embedding::text
        FROM "claim"
        WHERE "clusterId" IS NULL
    """)

    if not claims or len(claims) < 2:
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

    if len(vectors) < 2:
        return

    X = np.array(vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms

    # HDBSCAN — min_cluster_size=2 is the least restrictive
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X)

    # Diagnostics
    groups: Dict[int, List[Dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(label, [])
        groups[label].append({"id": ids[idx], "text": texts[idx]})

    noise_count = sum(1 for l in labels if l == -1)
    logger.info(f"[DIAG] Claims: {len(ids)} | Clusters: {len(groups)} | Noise: {noise_count}")

    # Log each cluster before merge
    for lbl, members in groups.items():
        logger.info(f"[DIAG] Pre-merge cluster {lbl}: {len(members)} claims — {members[0]['text'][:60]}...")

    if not groups:
        logger.info("No clusters formed.")
        return

    # LLM Merge Pass (1 call)
    groups = _llm_merge_pass(groups)
    logger.info(f"[DIAG] After merge: {len(groups)} clusters.")

    # Per-cluster processing
    for label, members in groups.items():
        raw_claim_texts = [m["text"] for m in members]

        # Canonical claim lives on CLUSTER (Issue 2 fix)
        canonical = _generate_canonical_claim(raw_claim_texts)

        # Title via TF-IDF (Issue 8)
        title = generate_event_title_tfidf(raw_claim_texts)

        logger.info(f"[DIAG] Cluster → title='{title}' | canonical='{canonical[:60]}...' | claims={len(members)}")

        # Create cluster with canonical claim ON THE CLUSTER
        cluster_record = await prisma.claimcluster.create(
            data={
                "title": title,
                "canonicalClaim": canonical,
            }
        )

        # Issue 1+2 FIX: Only set clusterId on claims.
        # DO NOT overwrite canonicalClaim on individual claims.
        # Claims keep their original raw text.
        for member in members:
            await prisma.claim.update(
                where={"id": member["id"]},
                data={"clusterId": cluster_record.id},
            )

    logger.info("=== Clustering Complete ===")

# ── LLM Merge Pass ───────────────────────────────────────────────

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
        "Merge clusters describing the SAME real-world event. "
        "Only merge if claims are about the exact same story. "
        "Do NOT merge loosely related topics. "
        "Return JSON: "
        '{"merge_groups": [[id1, id2]]} '
        "If no merges: "
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

# ── Canonical Claim (per-cluster) ────────────────────────────────

def _generate_canonical_claim(claim_texts: List[str]) -> str:
    if len(claim_texts) == 1:
        return claim_texts[0]

    system_prompt = (
        "Canonicalize these related claims into ONE definitive factual statement. "
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
    logger.info("=== Phase 2 Final — Event Detection ===")

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
    events_skipped = 0

    for cluster in clusters:
        if not cluster.claims:
            continue

        # Aggregate at CLUSTER level (Issue 3 fix)
        claim_count = len(cluster.claims)
        all_evidence = []
        for c in cluster.claims:
            all_evidence.extend(c.evidence)

        evidence_count = len(all_evidence)
        sources = set(e.source for e in all_evidence)
        source_count = len(sources)

        # Event Eligibility (Issue 4: lowered to allow more events)
        if source_count < 2 and claim_count < 2:
            events_skipped += 1
            continue

        if evidence_count < 2:
            events_skipped += 1
            continue

        # Cross-Source Consensus (Issue 7, 9)
        consensus_score = min(source_count / max(claim_count, 1), 1.0)

        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"consensusScore": consensus_score},
        )

        # Event Title (TF-IDF, Issue 8)
        raw_texts = [c.canonicalClaim for c in cluster.claims]
        event_title = generate_event_title_tfidf(raw_texts)

        # Event Summary (1 LLM call)
        evidence_sentences = list(set(e.sentence for e in all_evidence))[:8]
        summary_prompt = json.dumps({
            "title": event_title,
            "canonical_claim": getattr(cluster, 'canonicalClaim', '') or cluster.title,
            "evidence": evidence_sentences[:4],
            "sources": list(sources),
        })
        raw_summary = call_llm(
            "Write a 1-sentence news summary. Return JSON: "
            '{"summary": "..."}',
            summary_prompt,
            max_tokens=128,
        )
        event_summary = parse_json_safe(raw_summary).get("summary", "")

        # Importance Score (Issue 7)
        importance = (
            source_count * 0.30
            + source_count * 0.20   # publisher_diversity ≈ source_count
            + evidence_count * 0.15
            + claim_count * 0.15
            + consensus_score * 0.20
        )
        if source_count > 3:
            importance += 2.0

        # Store Event
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
        logger.info(f"[EVENT] '{event_title}' — sources={source_count} claims={claim_count} evidence={evidence_count} importance={importance:.2f}")

    logger.info(f"=== Event Detection: {events_created} created, {events_skipped} skipped ===")
