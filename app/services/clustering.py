import logging
import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List, Dict

logger = logging.getLogger(__name__)

async def run_claim_clustering(prisma):
    """
    Step 6: Claim Clustering
    Groups semantically related Canonical Claims into Claim Clusters.
    """
    logger.info("Starting Claim Clustering...")
    
    # We must fetch embeddings as text because prisma-client-py cannot native-parse vector types
    claims = await prisma.query_raw('''
        SELECT id, "canonicalClaim", "clusterId", embedding::text
        FROM "claim"
    ''')
    
    if not claims or len(claims) < 5:
        logger.info("Not enough claims to run HDBSCAN clustering.")
        return

    embeddings = []
    claim_ids = []
    
    for c in claims:
        # Parse '[0.1, 0.2, ...]' back to list of floats
        vector_str = c["embedding"]
        if vector_str.startswith("[") and vector_str.endswith("]"):
            vector_str = vector_str[1:-1]
            vector = [float(x) for x in vector_str.split(",")]
            embeddings.append(vector)
            claim_ids.append(c["id"])

    X = np.array(embeddings)
    
    # HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    labels = clusterer.fit_predict(X)
    
    # Group claims by their new cluster label
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1: # Noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(claim_ids[i])
        
    for label, ids in clusters.items():
        # Find if any of these claims already have a clusterId (Merge Strategy)
        existing_cluster_ids = []
        for c in claims:
            if c["id"] in ids and c["clusterId"]:
                existing_cluster_ids.append(c["clusterId"])
                
        if existing_cluster_ids:
            # Use the most common existing cluster ID (majority vote)
            target_cluster_id = max(set(existing_cluster_ids), key=existing_cluster_ids.count)
        else:
            # Create a new ClaimCluster
            new_cluster = await prisma.claimcluster.create(
                data={
                    "title": "Generated Cluster (Pending LLM Title)"
                }
            )
            target_cluster_id = new_cluster.id
            
        # Update all claims in this cluster
        for cid in ids:
            await prisma.claim.update(
                where={"id": cid},
                data={"clusterId": target_cluster_id}
            )
            
    logger.info(f"Clustering complete. Formed {len(clusters)} valid clusters.")

async def run_event_detection(prisma):
    """
    Step 7: Event Detection
    Groups Claim Clusters into overarching Events using a secondary layer of abstraction.
    """
    logger.info("Starting Event Detection...")
    
    # In a fully robust system, we would embed the ClaimCluster titles or centroids
    # and run HDBSCAN again. For Phase 2, we will fetch unassigned clusters and group them.
    clusters = await prisma.claimcluster.find_many(
        where={"eventId": None},
        include={"claims": True}
    )
    
    if not clusters:
        return
        
    # As a foundational placeholder for Phase 2, we'll map standalone clusters to distinct Events.
    # Future iterations will run BERTopic over the clustered claims to generate event names.
    for cluster in clusters:
        if not cluster.claims:
            continue
            
        first_claim = cluster.claims[0].canonicalClaim
        # A simple LLM pass should generate the title here
        event_title = f"Event related to: {first_claim[:50]}..."
        
        new_event = await prisma.event.create(
            data={
                "title": event_title,
                "description": "Auto-generated event spanning multiple semantic claims."
            }
        )
        
        await prisma.claimcluster.update(
            where={"id": cluster.id},
            data={"eventId": new_event.id}
        )
        
    logger.info("Event Detection complete.")
