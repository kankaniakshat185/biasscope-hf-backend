# ADR 001: Vector Database Selection (pgvector vs Pinecone)

**Date:** July 2026  
**Status:** Accepted  

## Context
BiasScope needs to perform Semantic Claim Clustering. This requires generating 384-dimensional embeddings for thousands of claims and performing cosine similarity searches (nearest neighbor) to group identical claims into canonical Events.

## Alternatives Considered
1. **Pinecone (Managed Vector Database):** An external SaaS solution dedicated purely to vector similarity search.
2. **PostgreSQL with `pgvector` Extension:** A standard relational database augmented with an extension to support vector data types and HNSW/IVFFlat indexes.
3. **ChromaDB / FAISS:** Local, in-memory vector stores.

## Decision
We selected **PostgreSQL with `pgvector`**.

## Justification
1. **Relational Coupling:** In BiasScope, vectors are not standalone documents. A `Claim` vector is tightly coupled to foreign keys (`Article ID`, `Search ID`, `Event ID`). If we used Pinecone, we would suffer from "Split-Brain Architecture"—having to sync relational metadata in Postgres with vectors in Pinecone. With `pgvector`, we can run a single SQL query: `SELECT * FROM "Claim" WHERE 1 - (embedding <=> vector) > 0.85 AND "articleId" = X`.
2. **Cost & Overhead:** Pinecone introduces an additional monthly SaaS cost and a new API dependency. `pgvector` runs inside our existing Prisma/PostgreSQL infrastructure at zero additional overhead.
3. **Performance:** Our dataset size per clustering run (thousands of claims) easily fits within the performant bounds of `pgvector`'s HNSW index, making a dedicated external vector database overkill.
