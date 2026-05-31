# BiasScope: Master Technical Audit & Expansion Plan

This document serves as the ultimate architectural blueprint for evolving BiasScope from a basic sentiment dashboard into an **Evidence-Based Media Intelligence System**. The core architectural shift is moving from an **Article-Centric** pipeline to a **Claim-Centric** intelligence layer. 

By building advanced features like Contradiction Detection and Multi-Perspective RAG on top of extracted claims rather than raw articles, BiasScope transforms into a highly differentiated AI platform capable of detecting cross-ideological consensus and flagging narrative drift.

---

## I. Revised Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion
        A[Raw URL / Topic Subscription] --> B[Article Scraper]
        B --> C[Static Source Registry]
        C --> D[PoliticalBiasBERT NLP]
    end

    subgraph Phase 1: Trust & Metrics
        D --> E[Source vs Article Bias Deviation]
        D --> F[Dataset Metrics & Diversity]
    end

    subgraph Phase 2: Claim Extraction Foundation
        D --> G[LLM Claim Extraction Engine]
        G --> H[Claim Normalization]
        H --> I[Sentence-Transformers Clustering]
        I --> J[(Global Claim Database w/ pgvector)]
    end

    subgraph Phase 3: Consensus & Contradiction Intelligence
        J --> K[Evidence Linking]
        K --> L[NLI Model DeBERTa-v3]
        L --> M{Consensus vs Contradiction}
        M -->|Agreement| N[Cross-Ideological Consensus Facts]
        M -->|Conflict| O[Contested Claims]
    end

    subgraph Phase 3.5: Longitudinal Intelligence (Weekly Snapshots)
        N --> P[Topic Snapshot Generation Delta-Processing]
        O --> P
        P --> Q[Narrative Drift Detection]
    end

    subgraph Phase 4: Multi-Perspective RAG
        J --> R[Claim-Centric Retrieval]
        R --> S[Ideology-Balanced Context Injection]
        S --> T[Contradiction-Aware RAG Synthesizer]
    end

    subgraph Phase 7: UI & Visualization
        P --> U[Research Analyst Dashboard]
        T --> U
        U --> V[/demo Route Read-Only Fast Load]
    end
```

---

## II. Updated Prisma Schema Proposal

This schema drops the localized `Search` approach. `Claim` and `Evidence` become global, first-class entities.

```prisma
// Claims are global intelligence assets, not scoped to individual searches.
model Claim {
  id                  String     @id @default(uuid())
  canonical_statement String     @unique
  embedding           Unsupported("vector(384)") // Requires pgvector extension
  createdAt           DateTime   @default(now())
  lastUpdatedAt       DateTime   @updatedAt
  
  // Evolving Intelligence Scores
  consensusScore      Float      @default(0.0)
  contradictionScore  Float      @default(0.0)

  evidence            Evidence[]
  topicSnapshots      TopicSnapshotClaim[]
}

model Evidence {
  id              String   @id @default(uuid())
  claimId         String
  articleId       String
  
  exactSentence   String    // Grounding for RAG/UI
  sourceBias      String    // The ideological leaning of the source
  nliRelationship String    // "SUPPORT", "CONTRADICT", "NEUTRAL"
  timestamp       DateTime  @default(now())
  
  claim           Claim     @relation(fields: [claimId], references: [id], onDelete: Cascade)
  article         Article   @relation(fields: [articleId], references: [id], onDelete: Cascade)
}

// Longitudinal Intelligence
model TopicSubscription {
  id              String   @id @default(uuid())
  topic           String   @unique
  isActive        Boolean  @default(true)
  snapshots       TopicSnapshot[]
}

model TopicSnapshot {
  id              String   @id @default(uuid())
  subscriptionId  String
  weekNumber      Int
  narrativeDrift  Float?
  createdAt       DateTime @default(now())

  subscription    TopicSubscription @relation(fields: [subscriptionId], references: [id])
  claims          TopicSnapshotClaim[]
}
```

---

## III. Claim Lifecycle Pipeline

1. **Extraction:** LLM extracts declarative facts from an incoming article.
2. **Normalization:** Extracted claims are standardized (e.g., stripping passive voice).
3. **Clustering:** Embeddings are generated. A Cosine Similarity check against the global `Claim` table determines if it's a new claim or a duplicate.
4. **Evidence Linking:** The exact sentence is mapped to the matched `Claim` in the `Evidence` table.
5. **NLI Verification:** DeBERTa-v3 runs an NLI check to see if the new sentence SUPPORTS or CONTRADICTS the canonical claim.
6. **Consensus Update:** The `consensusScore` and `contradictionScore` on the `Claim` row are dynamically re-calculated based on the new evidence.

---

## IV. Weekly Topic Intelligence (Phase 3.5) & Resource Cost Analysis

**Architecture:** We avoid HuggingFace-local persistence and Real-Time synchronous scraping. Users subscribe to topics. A **Celery Worker** (backed by external Upstash Redis) triggers weekly. It fetches *only* articles published since the last snapshot (Delta Processing), updates the global Claim Clusters, and writes a new `TopicSnapshot` to PostgreSQL. 

### Resource Constraint Review (HuggingFace Free Tier + External DBs)
| Feature | CPU Cost | Memory Cost | Storage Cost | Feasibility on Free Tier |
|---------|----------|-------------|--------------|--------------------------|
| Claim Extraction (LLM API) | LOW | LOW | LOW | Yes (offloaded to API) |
| Sentence Embeddings (`MiniLM`) | MED | MED | MED (pgvector) | Yes |
| NLI Contradiction (`DeBERTa-v3`) | HIGH | HIGH | LOW | Yes, if batched async via Celery |
| Vector DB Retrieval (pgvector) | LOW | LOW | MED | Yes |
| Full SHAP Explainability | VERY HIGH | VERY HIGH | LOW | **NO (Defer to Phase 5 / Alternatives)** |
| High-Frequency Scraping | HIGH | LOW | HIGH | **NO (Use Weekly Snapshots instead)** |

---

## V. Final 7-Phase Implementation Order

### PHASE 1 — TRUST, CORRECTNESS, AND METRICS (Tier 1)
*Goal: Fix anything that damages trust before building complex intelligence features.*
* **Dependencies:** None
1. **Source Bias vs Article Bias Architecture:** Store both static `source_bias` and predicted `article_bias`. Compute a `deviation_score` to detect **Narrative Anomalies**.
2. **PoliticalBiasBERT Audit:** Rigorous dataset/label audit against Reuters, BBC, Fox, CNN. Deliver formal confidence assessment.
3. **Fix Confidence Metric:** Replace softmax average with `Source Reliability Confidence`.
4. **Fix Narrative Summary Overclaiming:** Hardcode boundaries in the prompt (*"Among the analyzed articles..."*).
5. **Fix Polarization Index:** Implement Jensen-Shannon Divergence.
6. **Dataset Metrics:** Implement Source Diversity, Geographic Diversity, and Coverage Imbalance metrics.

### PHASE 2 — CLAIM EXTRACTION FOUNDATION (Tier 1)
*Goal: Create the global intelligence layer that everything else will depend on.*
* **Dependencies:** Phase 1
1. **Claim Extraction Engine:** Extract factual assertions from every article.
2. **Global Evidence Storage System:** Implement the proposed Prisma Schema.
3. **Claim Clustering:** Use `sentence-transformers` via `pgvector` to group semantically similar claims.
4. **Event Detection:** Use `BERTopic` / `HDBSCAN` to identify specific events driving the coverage.

### PHASE 3 — CONSENSUS & CONTRADICTION INTELLIGENCE (Tier 2)
*Goal: Transform claims into actionable intelligence. This is the flagship differentiator.*
* **Dependencies:** Phase 2
1. **Contradiction Detection Engine:** Use `DeBERTa-v3 NLI` to detect contradiction vs. support between evidence and claims.
2. **Consensus Detection Engine:** Flag claims supported across Left, Center, and Right sources. 
3. **Research Mode Backend:** Format API responses as intelligence briefings (Consensus Facts, Contested Claims, Missing Info).

### PHASE 3.5 — LONGITUDINAL INTELLIGENCE (Tier 3)
*Goal: Move from real-time synchronous analysis to asynchronous weekly intelligence tracking.*
* **Dependencies:** Phase 2, Phase 3
1. **Topic Subscription Architecture:** Implement Celery + Upstash Redis for weekly cron jobs.
2. **Snapshot Delta Processing:** Only process new articles published since the last snapshot.
3. **Narrative Drift Detection:** Track the evolution of Consensus and Contradiction scores over time.

### PHASE 4 — MULTI-PERSPECTIVE RAG (Tier 3)
*Goal: Build RAG on top of structured intelligence data rather than raw text dumps.*
* **Dependencies:** Phase 2, Phase 3
1. **Claim-Centric Retrieval:** Operate retrieval over `Claims` rather than raw chunks to massively improve explainability.
2. **Multi-Perspective Retrieval:** Force the retrieval system to pull independently from Left, Center, and Right clusters before synthesis.
3. **Contradiction-Aware RAG:** If retrieved sources disagree, the LLM must explicitly state: *"Sources disagree regarding this event."*
4. **Citation Enforcement:** Every response must cite the source, claim, and evidence sentence.

### PHASE 5 — EXPLAINABILITY & EVALUATION (Tier 4)
*Goal: Make the system auditable and mathematically rigorous.*
* **Dependencies:** Phase 1, Phase 4
1. **Lightweight Explainability:** Defer heavy SHAP processing. Use Lightweight Token Attribution or Attention Attribution for classification endpoints.
2. **Evaluation Framework:** Implement strict tracking for Bias Eval, Claim Extraction Eval (Precision/Recall), NLI Contradiction Eval, and RAG Eval.

### PHASE 7 — VISUALIZATION, POLISH, AND DEMO MODE (Tier 5)
*Goal: UI/UX transformation.*
* **Dependencies:** Phase 3.5, Phase 4
1. **Demo Mode:** Implement a `/demo` route. Pre-load a highly populated historical `TopicSnapshot` (e.g., "Elon Musk"). This will load instantly (read-only) and demonstrate Claim Extraction, Consensus, and Contradiction within 30 seconds for recruiters.
2. **Research Analyst Dashboard:** Restructure the UI into the multi-page intelligence briefing (Executive Summary, Consensus vs Disagreement, Entity Intelligence, Source Intelligence).
