"""
Phase 2 — Clustering & Intelligence Layer (SMART)

This is the brain of the intelligence pipeline.
It runs AFTER all claims have been extracted and stored.

Pipeline:
  1. Load all claims + embeddings
  2. Normalize embeddings
  3. HDBSCAN clustering
  4. LLM cluster merge pass (merge semantically identical clusters)
  5. Generate ONE canonical claim per cluster (LLM)
  6. Generate cluster title (LLM)
  7. Event eligibility check
  8. Event title + summary generation (LLM)
  9. Event importance ranking
  10. Store everything
"""

import os
import json
import logging
import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ── LLM Helper ───────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("No HF_TOKEN found for clustering LLM pass.")
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
    """Safely parse JSON from LLM output."""
    if not raw:
        return fallback or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback or {}

# ── Step 1: Clustering ───────────────────────────────────────────

async def run_claim_clustering(prisma):
    """
    Full intelligence pipeline:
      1. Load claims
      2. HDBSCAN on normalized embeddings
      3. LLM merge pass
      4. Canonical claim generation per cluster
      5. Cluster title generation
      6. Store clusters
    """
    logger.info("=== Starting Claim Clustering Pipeline ===")

    # 1. Load all unclustered claims
    claims = await prisma.query_raw("""
        SELECT id, "canonicalClaim", embedding::text
        FROM "claim"
    """)

    if not claims or len(claims) < 3:
        logger.info(f"Only {len(claims) if claims else 0} claims — skipping clustering.")
        return

    # Parse embeddings
    ids, texts, vectors = [], [], []
    for c in claims:
        vec_str = c["embedding"]
        try:
            vec = [float(x) for x in vec_str[1:-1].split(",")]
            vectors.append(vec)
            ids.append(c["id"])
            texts.append(c["canonicalClaim"])
        except (ValueError, IndexError):
            continue

    if len(vectors) < 3:
        return

    X = np.array(vectors)

    # 2. Normalize for cosine-equivalent euclidean
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # prevent division by zero
    X = X / norms

    # 3. HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X)

    # Group claims by cluster label
    groups: Dict[int, List[Dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # noise
            continue
        groups.setdefault(label, [])
        groups[label].append({"id": ids[idx], "text": texts[idx]})

    if not groups:
        logger.info("No clusters formed by HDBSCAN.")
        return

    logger.info(f"HDBSCAN produced {len(groups)} initial clusters from {len(ids)} claims.")

    # 4. LLM Cluster Merge Pass
    groups = await _llm_merge_pass(groups)
    logger.info(f"After LLM merge: {len(groups)} clusters.")

    # 5. For each cluster: canonicalize, title, store
    for label, members in groups.items():
        claim_texts = [m["text"] for m in members]

        # Generate canonical claim
        canonical_claim = _generate_canonical_claim(claim_texts)

        # Generate cluster title
        cluster_title = _generate_cluster_title(claim_texts, canonical_claim)

        # Create the ClaimCluster record
        cluster_record = await prisma.claimcluster.create(
            data={"title": cluster_title}
        )

        # Update all claims: assign to cluster + set canonical text
        for member in members:
            await prisma.claim.update(
                where={"id": member["id"]},
                data={
                    "clusterId": cluster_record.id,
                    "canonicalClaim": canonical_claim,
                },
            )

    logger.info("=== Clustering Pipeline Complete ===")

# ── Step 2: LLM Merge Pass ──────────────────────────────────────

async def _llm_merge_pass(groups: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    """Ask LLM which clusters represent the same real-world event."""
    if len(groups) <= 1:
        return groups

    cluster_payload = []
    label_list = list(groups.keys())
    for lbl in label_list:
        cluster_payload.append({
            "cluster_id": int(lbl),
            "claims": [m["text"] for m in groups[lbl]][:10],  # cap context
        })

    system_prompt = (
        "You are merging redundant claim clusters. "
        "Clusters that describe the SAME real-world event or story must be merged. "
        "Return ONLY valid JSON: "
        '{"merge_groups": [[id1, id2], [id3, id4, id5]]}'
        " If no merges are needed, return: "
        '{"merge_groups": []}'
    )
    user_prompt = f"Clusters:\n{json.dumps(cluster_payload)}"

    raw = call_llm(system_prompt, user_prompt)
    data = parse_json_safe(raw, {"merge_groups": []})
    merge_groups = data.get("merge_groups", [])

    for group in merge_groups:
        if not group or len(group) < 2:
            continue
        target = group[0]
        for src in group[1:]:
            if src in groups and target in groups:
                groups[target].extend(groups[src])
                del groups[src]

    return groups

# ── Canonical Claim Generation ───────────────────────────────────

def _generate_canonical_claim(claim_texts: List[str]) -> str:
    """Generate ONE canonical claim that summarizes all equivalent claims."""
    if len(claim_texts) == 1:
        return claim_texts[0]

    system_prompt = (
        "You are canonicalizing a cluster of related claims into ONE definitive claim. "
        "The canonical claim must be factual, complete, and self-contained. "
        "Return ONLY valid JSON: "
        '{"canonical_claim": "..."}'
    )
    user_prompt = json.dumps(claim_texts[:15])  # cap context

    raw = call_llm(system_prompt, user_prompt, max_tokens=256)
    data = parse_json_safe(raw)
    return data.get("canonical_claim", claim_texts[0])

# ── Cluster Title Generation ────────────────────────────────────

def _generate_cluster_title(claim_texts: List[str], canonical_claim: str) -> str:
    """Generate a short factual label for the cluster."""
    system_prompt = (
        "Generate a short factual label for this claim cluster. "
        "The label must be 3-10 words describing the real-world fact. "
        "Return ONLY valid JSON: "
        '{"title": "..."}'
    )
    user_prompt = json.dumps({
        "canonical_claim": canonical_claim,
        "supporting_claims": claim_texts[:10],
    })

    raw = call_llm(system_prompt, user_prompt, max_tokens=128)
    data = parse_json_safe(raw)
    return data.get("title", canonical_claim[:80])

# ── Step 3: Event Detection ─────────────────────────────────────

async def run_event_detection(prisma):
    """
    Event generation pipeline:
      1. Load all clusters with claims + evidence
      2. Check eligibility (multi-source, multi-claim, multi-evidence)
      3. Generate event title + summary via LLM
      4. Compute importance score
      5. Store events
    """
    logger.info("=== Starting Event Detection Pipeline ===")

    clusters = await prisma.claimcluster.find_many(
        where={"eventId": None},
        include={
            "claims": {
                "include": {"evidence": True}
            }
        },
    )

    if not clusters:
        logger.info("No unassigned clusters found.")
        return

    events_created = 0

    for cluster in clusters:
        if not cluster.claims:
            continue

        # Gather stats
        claim_count = len(cluster.claims)
        all_evidence = []
        for c in cluster.claims:
            all_evidence.extend(c.evidence)

        evidence_count = len(all_evidence)
        sources = set(e.source for e in all_evidence)
        source_count = len(sources)

        # ── Event Eligibility Rules ──
        if source_count < 2 or claim_count < 2 or evidence_count < 3:
            logger.debug(
                f"Cluster {cluster.id} ineligible: "
                f"sources={source_count}, claims={claim_count}, evidence={evidence_count}"
            )
            continue

        # ── Publisher Diversity ──
        publisher_diversity = source_count  # unique publisher domains

        # ── Event Title + Summary via LLM ──
        evidence_sentences = list(set(e.sentence for e in all_evidence))[:10]

        payload = {
            "canonical_claim": cluster.claims[0].canonicalClaim,
            "supporting_claims": [c.canonicalClaim for c in cluster.claims][:20],
            "evidence_sentences": evidence_sentences,
            "source_names": list(sources),
            "source_count": source_count,
            "evidence_count": evidence_count,
        }

        system_prompt = (
            "You are an expert news editor. Transform this claim cluster into a concise Event.\n"
            "RULES:\n"
            "1. event_title: 3–8 words, like a news headline. NOT a full sentence.\n"
            "2. event_title MUST NOT begin with 'Event related to'.\n"
            "3. event_summary: 1 sentence overview.\n"
            "4. event_category: one of [Politics, Business, Technology, Science, Culture, Legal, Other].\n"
            "Return ONLY valid JSON:\n"
            '{"event_title":"...","event_summary":"...","event_category":"..."}'
        )
        user_prompt = f"Cluster:\n{json.dumps(payload)}"

        raw = call_llm(system_prompt, user_prompt)
        event_data = parse_json_safe(raw, {
            "event_title": cluster.title,
            "event_summary": "",
            "event_category": "Other",
        })

        event_title = event_data.get("event_title", cluster.title)
        event_summary = event_data.get("event_summary", "")

        # Reject bad titles
        if event_title.lower().startswith("event related to"):
            event_title = cluster.title

        # ── Importance Score ──
        # importance = source_count*0.30 + publisher_diversity*0.20 + evidence_count*0.15 + claim_count*0.15 + consensus*0.20
        # consensus is approximated as publisher_diversity / max(source_count, 1) for now
        consensus_approx = min(publisher_diversity / max(source_count, 1), 1.0)
        importance = (
            source_count * 0.30
            + publisher_diversity * 0.20
            + evidence_count * 0.15
            + claim_count * 0.15
            + consensus_approx * 0.20
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

    logger.info(f"=== Event Detection Complete: {events_created} events created ===")
