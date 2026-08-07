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

import json
import logging
import re
import warnings
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from app.services.llm_client import cached_llm_call

# --- NLI Pipeline ---
# Use a fast DeBERTa v3 small model for NLI Contradiction routing
# Three states live in this one variable: None ("not attempted yet"),
# False ("attempted and failed — don't retry"), or a loaded pipeline.
# Typed Any rather than Optional[Pipeline] so the False sentinel doesn't
# need its own union member.
nli_classifier: Any = None
def get_nli_classifier():
    global nli_classifier
    if nli_classifier is None:
        import logging

        from transformers import pipeline
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
COHESION_THRESHOLD = 0.72

def parse_json_safe(raw: str, fallback: dict | None = None) -> dict:
    if not raw:
        return fallback or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback or {}

# ── Deterministic Event Title (0 LLM calls) ──────────────────────

def generate_event_title(claim_texts: list[str]) -> str:
    """
    Generate a descriptive event title using entity extraction + TF-IDF keywords.
    Handles both Western and non-Western entities, acronyms (BJP, GDP, SEC),
    and international political actions.
    """
    if not claim_texts:
        return "Unclassified Event"

    all_text = " ".join(claim_texts)

    # Capture CamelCase entities AND all-caps acronyms (BJP, RSS, GDP, IMF, SEC)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', all_text)
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', all_text)
    entity_counts = Counter(entities + acronyms)

    # Extended stopwords including common non-discriminating terms
    stopwords = {
        "The", "This", "That", "These", "Those", "According", "However",
        "While", "After", "Before", "During", "Between", "About", "Also",
        "Mr", "Mrs", "Ms", "Dr", "Inc", "Ltd", "Corp", "It", "He", "She",
        "Its", "His", "Her", "They", "Their", "New", "First",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
        "PM", "AM", "US", "UK", "IN", "AN", "OR", "AS", "AT", "TO", "OF",
    }
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

    # Expanded action map (financial + political + international)
    action_map = {
        # Financial / Legal
        "filed": "Filing", "ipo": "IPO", "merger": "Merger", "acquisition": "Acquisition",
        "lawsuit": "Lawsuit", "sued": "Lawsuit", "launch": "Launch", "launched": "Launch",
        "agreement": "Agreement", "deal": "Deal", "partnership": "Partnership",
        "compute": "Compute Deal", "signed": "Agreement", "controversy": "Controversy",
        "resign": "Resignation", "appointed": "Appointment", "invest": "Investment",
        "raised": "Funding", "acquired": "Acquisition", "bankruptcy": "Bankruptcy",
        "fraud": "Fraud", "regulation": "Regulation", "ban": "Ban",
        "explosion": "Explosion", "failure": "Failure", "crash": "Crash",
        "loss": "Financial Loss", "revenue": "Revenue", "valuation": "Valuation",
        "trading": "Trading", "shares": "Share Offering", "billion": "Financial",
        "convicted": "Conviction", "arrested": "Arrest", "murder": "Murder",
        "lease": "Lease Agreement", "data center": "Data Center",
        "satellite": "Satellite", "rocket": "Rocket", "test": "Test",
        "settlement": "Settlement", "penalty": "Penalty", "sec": "SEC Action",
        "trillionaire": "Wealth", "net worth": "Valuation",
        # Political / International
        "election": "Election", "vote": "Vote", "campaign": "Campaign",
        "rally": "Rally", "protest": "Protest", "summit": "Summit",
        "sanctions": "Sanctions", "tariff": "Tariff", "treaty": "Treaty",
        "cabinet": "Cabinet Reshuffle", "sworn": "Inauguration", "oath": "Inauguration",
        "manifesto": "Manifesto", "alliance": "Alliance", "coalition": "Coalition",
        "ceasefire": "Ceasefire", "war": "Conflict", "attack": "Attack",
        "speech": "Speech", "address": "Address", "visit": "State Visit",
        "bilateral": "Bilateral Talks", "trade": "Trade", "policy": "Policy",
        "reform": "Reform", "budget": "Budget", "infrastructure": "Infrastructure",
        "defense": "Defense", "defence": "Defence", "military": "Military",
        "nuclear": "Nuclear", "missile": "Missile", "drone": "Drone Strike",
        "deportation": "Deportation", "immigration": "Immigration",
        "climate": "Climate", "pandemic": "Pandemic", "vaccine": "Vaccine",
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
        title = f"{top_entities[0]} {action_word}"
    elif top_entities and top_keywords:
        kw_text = top_keywords[0].title()
        if kw_text.lower() not in top_entities[0].lower():
            title = f"{top_entities[0]}: {kw_text}"
        else:
            fallback_kw = top_keywords[1].title() if len(top_keywords) > 1 else "Development"
            title = f"{top_entities[0]} {fallback_kw}"
    elif top_entities:
        title = " ".join(top_entities[:2])
    else:
        # Last resort: truncate first claim intelligently
        first = claim_texts[0]
        title = first[:70].rsplit(" ", 1)[0] if len(first) > 70 else first

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


# Every /search used to trigger a full reclustering pass over every
# unclustered claim in the entire database, regardless of topic — so
# searching "elon musk" would pay the O(n^2) cohesion-comparison cost of
# reclustering leftover unclustered claims from someone else's unrelated
# "monsoon forecast" search too. Scoping to the current query fixes both
# the cost (bounded by one topic's claim volume, not the whole table) and
# a correctness issue (claims about unrelated topics were never usefully
# comparable anyway). Cross-search consensus for the SAME topic is
# preserved — claims from an earlier search or a weekly snapshot re-run of
# "elon musk" still join this query's claims, since they share the same
# search.query all the way through the evidence -> article -> search chain.
#
# `query=None` keeps the old table-wide behavior, used only by the
# /debug/rerun-* admin tools that intentionally rebuild every cluster from
# scratch after wiping the cluster table.
async def run_claim_clustering(prisma, query: str | None = None):
    logger.info(f"=== Clustering Pipeline (query={query!r}) ===")

    if query:
        claims = await prisma.query_raw(
            """
            SELECT DISTINCT c.id, c."canonicalClaim", c.embedding::text
            FROM "claim" c
            JOIN "evidence" e ON e."claimId" = c.id
            JOIN "article" a ON a.id = e."articleId"
            JOIN "search" s ON s.id = a."searchId"
            WHERE c."clusterId" IS NULL
              AND LOWER(s.query) = LOWER($1)
            """,
            query,
        )
    else:
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
        min_samples=1,
        metric="precomputed",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.15,
    )
    labels = clusterer.fit_predict(cos_dist)

    groups: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(label, [])
        groups[label].append({"id": ids[idx], "text": texts[idx], "vec_idx": idx})

    noise_count = sum(1 for lbl in labels if lbl == -1)
    logger.info(f"[DIAG] Claims: {len(ids)} | Clusters: {len(groups)} | Noise: {noise_count}")

    if not groups:
        return

    # ── Cohesion Validation ──
    # Reject clusters where claims are topically related but not about the same event.
    cohesive_groups: dict[int, dict[str, Any]] = {}
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
    for _label, data in cohesive_groups.items():
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

async def _generate_canonical_claim(prisma, claim_texts: list[str]) -> str:
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
        # Consensus = what fraction of claims are corroborated by 2+ distinct sources
        if claim_count > 0:
            corroborated_claims = 0
            for c in cluster.claims:
                claim_sources = set(e.source for e in c.evidence)
                if len(claim_sources) >= 2:
                    corroborated_claims += 1
            consensus_score = corroborated_claims / claim_count
        else:
            consensus_score = 0.0
        # Bonus for source diversity
        source_diversity_bonus = min(source_count / 5.0, 0.3)
        consensus_score = min(consensus_score + source_diversity_bonus, 1.0)

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

        # ── Importance Score (normalized 0-1) ──
        importance = (
            min(source_count / 10.0, 1.0) * 0.30 +
            publisher_diversity * 0.15 +
            min(evidence_count / 15.0, 1.0) * 0.20 +
            min(claim_count / 8.0, 1.0) * 0.15 +
            consensus_score * 0.20
        )

        # Bonus for cross-source coverage (normalized)
        if source_count >= 5:
            importance = min(importance + 0.15, 1.0)
        elif source_count >= 3:
            importance = min(importance + 0.08, 1.0)

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
