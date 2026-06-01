import logging
import os
import json
import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found for clustering LLM pass.")
        return ""
    
    # We use Meta-Llama-3-8B-Instruct or Qwen2.5
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model=model_id, token=hf_token)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
        content = response.choices[0].message.content.strip()
        # Clean markdown
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

async def run_claim_clustering(prisma):
    """
    Step 6: Claim Clustering (with LLM Merge Pass - Issue 2)
    """
    logger.info("Starting Claim Clustering...")
    
    claims = await prisma.query_raw('''
        SELECT id, "canonicalClaim", "clusterId", embedding::text
        FROM "claim"
    ''')
    
    if not claims or len(claims) < 5:
        logger.info("Not enough claims to run HDBSCAN clustering.")
        return

    embeddings = []
    claim_ids = []
    claim_texts = []
    
    for c in claims:
        vector_str = c["embedding"]
        if vector_str.startswith("[") and vector_str.endswith("]"):
            vector_str = vector_str[1:-1]
            vector = [float(x) for x in vector_str.split(",")]
            embeddings.append(vector)
            claim_ids.append(c["id"])
            claim_texts.append(c["canonicalClaim"])

    X = np.array(embeddings)
    
    # 1. Initial HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    labels = clusterer.fit_predict(X)
    
    clusters_map = {}
    for i, label in enumerate(labels):
        if label == -1: 
            continue
        if label not in clusters_map:
            clusters_map[label] = []
        clusters_map[label].append({"id": claim_ids[i], "text": claim_texts[i]})
        
    if not clusters_map:
        return
        
    # 2. LLM Cluster Merge Pass (Issue 2)
    cluster_payload = []
    for lbl, members in clusters_map.items():
        cluster_payload.append({
            "cluster_id": int(lbl),
            "claims": [m["text"] for m in members]
        })
        
    system_prompt = (
        "You are an AI tasked with merging redundant event clusters. "
        "Review the provided JSON array of claim clusters. "
        "Determine which clusters represent the exact same real-world event/story and should be merged. "
        "Return ONLY a valid JSON object matching this schema: "
        '{"merge_groups": [[cluster_id_1, cluster_id_2], [cluster_id_3, cluster_id_4, cluster_id_5]]}'
    )
    user_prompt = f"Clusters:\n{json.dumps(cluster_payload)}"
    
    merge_resp = call_llm(system_prompt, user_prompt)
    merge_groups = []
    if merge_resp:
        try:
            merge_data = json.loads(merge_resp)
            merge_groups = merge_data.get("merge_groups", [])
        except json.JSONDecodeError:
            pass

    # Apply merges to labels
    for group in merge_groups:
        if not group or len(group) < 2: continue
        target_label = group[0]
        for src_label in group[1:]:
            if src_label in clusters_map and target_label in clusters_map:
                clusters_map[target_label].extend(clusters_map[src_label])
                del clusters_map[src_label]

    # 3. Save to DB
    for label, members in clusters_map.items():
        ids = [m["id"] for m in members]
        existing_cluster_ids = []
        for c in claims:
            if c["id"] in ids and c["clusterId"]:
                existing_cluster_ids.append(c["clusterId"])
                
        if existing_cluster_ids:
            target_cluster_id = max(set(existing_cluster_ids), key=existing_cluster_ids.count)
        else:
            new_cluster = await prisma.claimcluster.create(data={"title": members[0]["text"][:50]})
            target_cluster_id = new_cluster.id
            
        for cid in ids:
            await prisma.claim.update(
                where={"id": cid},
                data={"clusterId": target_cluster_id}
            )
            
    logger.info("Clustering and LLM Merge Pass complete.")

async def run_event_detection(prisma):
    """
    Step 7: Event Detection
    Filters eligible clusters, generates titles/summaries (Issue 1, 3), and ranks importance (Issue 4).
    """
    logger.info("Starting Event Detection...")
    
    clusters = await prisma.claimcluster.find_many(
        where={"eventId": None},
        include={
            "claims": {
                "include": {"evidence": True}
            }
        }
    )
    
    if not clusters:
        return
        
    for cluster in clusters:
        if not cluster.claims:
            continue
            
        # Calculate stats for Eligibility Rules
        claim_count = len(cluster.claims)
        all_evidence = []
        for c in cluster.claims:
            all_evidence.extend(c.evidence)
            
        evidence_count = len(all_evidence)
        sources = set([e.source for e in all_evidence])
        source_count = len(sources)
        
        # Issue 3: Event Eligibility Rules
        if not (source_count >= 2 and claim_count >= 2 and evidence_count >= 3):
            logger.info(f"Cluster {cluster.id} ineligible for Event status (sources={source_count}, claims={claim_count}, evidence={evidence_count})")
            continue
            
        # Issue 1: Event Title & Summary Generation
        payload = {
            "canonical_claim": cluster.claims[0].canonicalClaim,
            "supporting_claims": [c.canonicalClaim for c in cluster.claims],
            "source_count": source_count,
            "evidence_count": evidence_count
        }
        
        system_prompt = (
            "You are an expert news editor. Your job is to transform a cluster of claims into a concise, high-level Event Title and Summary. "
            "RULES:\n"
            "1. The Title MUST be 3-8 words and represent the underlying real-world story (e.g., 'SpaceX-Anthropic Compute Partnership').\n"
            "2. The Title MUST NOT be a full sentence, and MUST NOT begin with 'Event related to'.\n"
            "3. The Summary should be a 1-sentence overview of the event.\n"
            "Return ONLY valid JSON matching: "
            '{"event_title": "...", "event_summary": "..."}'
        )
        user_prompt = f"Cluster details:\n{json.dumps(payload)}"
        
        event_resp = call_llm(system_prompt, user_prompt)
        # Default fallback if LLM fails
        event_title = "Unclassified Claim Cluster"
        event_summary = "Auto-generated event cluster pending naming."
        
        if event_resp:
            try:
                data = json.loads(event_resp)
                event_title = data.get("event_title", event_title)
                event_summary = data.get("event_summary", event_summary)
            except:
                pass
                
        # Issue 4: Event Importance Scoring
        # importance = (source_count * 0.35 + evidence_count * 0.25 + publisher_diversity * 0.20 + claim_count * 0.20)
        # We approximate publisher_diversity as source_count for this metric
        importance = (source_count * 0.35) + (evidence_count * 0.25) + (source_count * 0.20) + (claim_count * 0.20)
        if source_count > 3:
            importance += 2.0 # Publisher diversity boost
            
        new_event = await prisma.event.create(
            data={
                "title": event_title,
                "description": event_summary,
                "importanceScore": importance
            }
        )
        
        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"eventId": new_event.id}
        )
        
    logger.info("Event Detection complete.")
