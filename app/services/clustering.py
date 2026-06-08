"""
Phase 2 Final — Clustering & Event Detection

Pipeline:
  1. Load unclustered claims + embeddings
  2. Cosine distance matrix → HDBSCAN (leaf selection)
  3. Event cohesion validation (pairwise cosine sim threshold)
  4. Canonical claim per cluster (1 cached LLM call each)
  5. Deterministic event title (TF-IDF + NER + action mapping)
  6. Event eligibility gate: sources >= 2 AND claims >= 2 AND evidence >= 2
  7. Cross-source consensus scoring
  8. Weighted importance ranking

All LLM calls go through llm_client.py for caching + analytics.
"""

import re
import json
import logging
import numpy as np
from collections import Counter
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from typing import List, Dict, Any
from app.services.llm_client import cached_llm_call
import warnings

# --- NLI Pipeline ---
# Use a fast DeBERTa v3 small model for NLI Contradiction routing
nli_classifier = None
def get_nli_classifier():
    global nli_classifier
    if nli_classifier is None:
        from transformers import pipeline
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            # cross-encoder/nli-deberta-v3-small is extremely fast and effective
            nli_classifier = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small", top_k=None)
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            nli_classifier = False
    return nli_classifier

logger = logging.getLogger(__name__)

# Minimum mean pairwise cosine similarity for a cluster to be considered
# a coherent event rather than a loose topic grouping.
COHESION_THRESHOLD = 0.65

def parse_json_safe(raw: str, fallback: dict = None) -> dict:
    if not raw:
        return fallback or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback or {}

# ── Deterministic Event Title (0 LLM calls) ──────────────────────

def generate_event_title(claim_texts: List[str]) -> str:
    """
    Generate a descriptive event title using entity extraction + TF-IDF keywords.
    Produces titles like "SpaceX IPO Filing" not "Elon Musk" or generic names.
    """
    if not claim_texts:
        return "Unclassified Event"

    all_text = " ".join(claim_texts)

    # Extract named entities (multi-word capitalized phrases)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', all_text)
    entity_counts = Counter(entities)

    # Remove generic stopword-like entities and single-word person titles
    stopwords = {"The", "This", "That", "These", "Those", "According", "However",
                 "While", "After", "Before", "During", "Between", "About", "Also",
                 "Mr", "Mrs", "Ms", "Dr", "Inc", "Ltd", "Corp", "It", "He", "She",
                 "Its", "His", "Her", "They", "Their", "New", "First", "Thursday",
                 "Friday", "Monday", "Tuesday", "Wednesday", "Saturday", "Sunday",
                 "January", "February", "March", "April", "May", "June", "July",
                 "August", "September", "October", "November", "December"}
    for sw in stopwords:
        entity_counts.pop(sw, None)

    top_entities = [e for e, _ in entity_counts.most_common(3)]

    # TF-IDF keywords
    try:
        vectorizer = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(claim_texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
    except Exception:
        top_keywords = []

    # Map keywords to action labels
    action_map = {
        "filed": "Filing", "ipo": "IPO", "merger": "Merger", "acquisition": "Acquisition",
        "lawsuit": "Lawsuit", "sued": "Lawsuit", "launch": "Launch", "launched": "Launch",
        "agreement": "Agreement", "deal": "Deal", "partnership": "Partnership",
        "compute": "Compute Deal", "signed": "Agreement", "controversy": "Controversy",
        "political": "Political", "election": "Election", "resign": "Resignation",
        "appointed": "Appointment", "invest": "Investment", "raised": "Funding",
        "acquired": "Acquisition", "bankruptcy": "Bankruptcy", "fraud": "Fraud",
        "safety": "Safety", "regulation": "Regulation", "ban": "Ban",
        "explosion": "Explosion", "failure": "Failure", "crash": "Crash",
        "loss": "Financial Loss", "revenue": "Revenue", "valuation": "Valuation",
        "trading": "Trading", "shares": "Share Offering", "billion": "Financial",
        "convicted": "Conviction", "arrested": "Arrest", "murder": "Murder",
        "lease": "Lease Agreement", "data center": "Data Center",
        "satellite": "Satellite", "rocket": "Rocket", "test": "Test",
        "settlement": "Settlement", "penalty": "Penalty", "sec": "SEC Action",
        "trillionaire": "Wealth", "net worth": "Valuation",
    }

    action_word = ""
    for kw in top_keywords:
        for trigger, label in action_map.items():
            if trigger in kw.lower():
                action_word = label
                break
        if action_word:
            break

    # Also check claim text directly for action words
    if not action_word:
        all_text_lower = all_text.lower()
        for trigger, label in action_map.items():
            if trigger in all_text_lower:
                action_word = label
                break

    # Build title — avoid repeating the same entity
    if top_entities and action_word:
        # Use most prominent entity + action
        title = f"{top_entities[0]} {action_word}"
    elif top_entities:
        # Use entity + first keyword as a fallback
        kw_text = top_keywords[0].title() if top_keywords else ""
        if kw_text and kw_text not in top_entities[0]:
            title = f"{top_entities[0]}: {kw_text}"
        else:
            title = " ".join(top_entities[:2])
    else:
        # Last resort: use truncated first claim
        title = claim_texts[0][:80]

    return title.strip()

# ── Event Cohesion Validation (no LLM) ───────────────────────────

def compute_cluster_cohesion(vectors: np.ndarray) -> float:
    """
    Compute mean pairwise cosine similarity within a cluster.
    Returns a value between 0 and 1.
    High cohesion = claims are about the same specific event.
    Low cohesion = claims are merely about the same topic/person.
    """
    if len(vectors) < 2:
        return 1.0
    sim_matrix = cos_sim(vectors)
    # Extract upper triangle (excluding diagonal)
    n = len(vectors)
    pair_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sims.append(sim_matrix[i][j])
    return float(np.mean(pair_sims)) if pair_sims else 1.0

# ══════════════════════════════════════════════════════════════════
# STEP 1: CLAIM CLUSTERING
# ══════════════════════════════════════════════════════════════════

async def run_claim_clustering(prisma):
    logger.info("=== Clustering Pipeline ===")

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

    # Cosine distance matrix for topic-aware clustering
    cos_dist = 1 - cos_sim(X)
    np.fill_diagonal(cos_dist, 0)

    clusterer = HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        metric="precomputed",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(cos_dist)

    groups: Dict[int, List[Dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(label, [])
        groups[label].append({"id": ids[idx], "text": texts[idx], "vec_idx": idx})

    noise_count = sum(1 for l in labels if l == -1)
    logger.info(f"[DIAG] Claims: {len(ids)} | Clusters: {len(groups)} | Noise: {noise_count}")

    if not groups:
        return

    # ── Cohesion Validation ──
    # Reject clusters where claims are topically related but not about the same event.
    cohesive_groups = {}
    rejected_cohesion = 0
    for label, members in groups.items():
        member_vecs = np.array([X[m["vec_idx"]] for m in members])
        cohesion = compute_cluster_cohesion(member_vecs)
        if cohesion >= COHESION_THRESHOLD:
            cohesive_groups[label] = {"members": members, "cohesion": cohesion}
        else:
            rejected_cohesion += 1
            logger.info(f"[COHESION] Cluster {label} rejected (cohesion={cohesion:.3f}, threshold={COHESION_THRESHOLD})")
            # Still persist rejected clusters for analysis (but they won't get events)
            raw_claim_texts = [m["text"] for m in members]
            title = generate_event_title(raw_claim_texts)
            cluster_record = await prisma.claimcluster.create(
                data={"title": title, "canonicalClaim": raw_claim_texts[0], "cohesionScore": cohesion}
            )
            for member in members:
                await prisma.claim.update(
                    where={"id": member["id"]},
                    data={"clusterId": cluster_record.id},
                )

    if rejected_cohesion > 0:
        logger.info(f"[DIAG] Cohesion gate: {rejected_cohesion} clusters rejected, {len(cohesive_groups)} passed")

    # Per-cluster: generate canonical claim + title, persist to DB
    for label, data in cohesive_groups.items():
        members = data["members"]
        cohesion = data["cohesion"]
        raw_claim_texts = [m["text"] for m in members]
        canonical = await _generate_canonical_claim(prisma, raw_claim_texts)
        title = generate_event_title(raw_claim_texts)

        cluster_record = await prisma.claimcluster.create(
            data={"title": title, "canonicalClaim": canonical, "cohesionScore": cohesion}
        )

        for member in members:
            await prisma.claim.update(
                where={"id": member["id"]},
                data={"clusterId": cluster_record.id},
            )

    logger.info("=== Clustering Complete ===")

# ── Canonical Claim (1 cached call per cluster) ──────────────────

async def _generate_canonical_claim(prisma, claim_texts: List[str]) -> str:
    if len(claim_texts) == 1:
        return claim_texts[0]

    system_prompt = (
        "Canonicalize these related claims into ONE definitive factual statement. "
        "Return JSON: "
        '{"canonical_claim": "..."}'
    )
    raw = await cached_llm_call(prisma, "canonicalization", system_prompt, json.dumps(claim_texts[:15]), max_tokens=256)
    data = parse_json_safe(raw)
    return data.get("canonical_claim", claim_texts[0])

# ══════════════════════════════════════════════════════════════════
# STEP 2: EVENT DETECTION (Quality Gate)
# ══════════════════════════════════════════════════════════════════

async def run_event_detection(prisma):
    logger.info("=== Event Detection (Quality Gate) ===")

    clusters = await prisma.claimcluster.find_many(
        where={"eventId": None},
        include={"claims": {"include": {"evidence": True}}},
    )

    if not clusters:
        return

    events_created = 0
    events_skipped = 0

    for cluster in clusters:
        if not cluster.claims:
            events_skipped += 1
            continue

        claim_count = len(cluster.claims)
        all_evidence = []
        for c in cluster.claims:
            all_evidence.extend(c.evidence)

        evidence_count = len(all_evidence)
        sources = set(e.source for e in all_evidence)
        source_count = len(sources)
        unique_urls = set(e.url for e in all_evidence)
        url_count = len(unique_urls)

        # ── Cross-Source Consensus & Polarization Score (via NLI) ──
        consensus_score = min(source_count / max(claim_count, 1), 1.0)
        
        # Publisher diversity
        publisher_diversity = min(source_count / max(url_count, 1), 1.0) if url_count > 0 else 0.0

        polarization_score = 0.0
        # If we have multiple unique claims, check for contradictions
        if claim_count >= 2:
            clf = get_nli_classifier()
            if clf:
                contradiction_count = 0
                total_pairs = 0
                # Take up to 5 claims to compare to avoid quadratic explosion
                claims_to_compare = [c.canonicalClaim for c in cluster.claims[:5]]
                for i in range(len(claims_to_compare)):
                    for j in range(i+1, len(claims_to_compare)):
                        pair_text = f"{claims_to_compare[i]} [SEP] {claims_to_compare[j]}"
                        try:
                            # Run zero-shot inference
                            res = clf(pair_text)
                            if res:
                                # the output is a list of dicts: [{'label': 'Contradiction', 'score': 0.99}, ...]
                                # check if Contradiction is the top label or has high score
                                top_label = res[0][0]['label'] if isinstance(res[0], list) else res[0]['label']
                                if top_label.lower() == 'contradiction':
                                    contradiction_count += 1
                        except Exception:
                            pass
                        total_pairs += 1
                if total_pairs > 0:
                    polarization_score = contradiction_count / total_pairs

        # If polarization is high, consensus drops!
        if polarization_score > 0.3:
            consensus_score = max(0.0, consensus_score - polarization_score)

        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"consensusScore": consensus_score},
        )


        # ── EVENT ELIGIBILITY GATE ──
        # An event must represent cross-source convergence.
        has_minimum_claims = claim_count >= 2
        has_minimum_evidence = evidence_count >= 2
        is_multi_source = source_count >= 2

        if not (has_minimum_claims and has_minimum_evidence and is_multi_source):
            events_skipped += 1
            continue

        # ── Importance Score (weighted) ──
        importance = (
            source_count * 0.30 +
            publisher_diversity * 0.20 +
            evidence_count * 0.15 +
            claim_count * 0.15 +
            consensus_score * 0.20
        )

        # Bonus for strong cross-source coverage
        if source_count >= 3:
            importance += 1.5
        if source_count >= 5:
            importance += 2.0

        # ── Event Title ──
        raw_texts = [c.canonicalClaim for c in cluster.claims]
        event_title = generate_event_title(raw_texts)

        # ── Event Summary (1 cached LLM call) ──
        summary_prompt = json.dumps({
            "title": event_title,
            "canonical_claim": getattr(cluster, 'canonicalClaim', '') or cluster.title,
            "evidence": list(set(e.sentence for e in all_evidence))[:4],
            "sources": list(sources),
        })
        raw_summary = await cached_llm_call(
            prisma, "event_summary",
            'Write a 1-sentence factual news summary. Do NOT include opinions. Return JSON: {"summary": "..."}',
            summary_prompt, max_tokens=128,
        )
        event_summary = parse_json_safe(raw_summary).get("summary", "")

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
        logger.info(
            f"[EVENT] '{event_title}' — sources={source_count} claims={claim_count} "
            f"evidence={evidence_count} consensus={consensus_score:.2f} importance={importance:.2f}"
        )

    logger.info(f"=== Events: {events_created} created, {events_skipped} skipped ===")
