# ADR 003: LLM Selection (Llama 3 8B vs GPT-4o)

**Date:** July 2026  
**Status:** Accepted  

## Context
BiasScope requires an LLM to parse thousands of raw news articles, extract atomic claims in a strict JSON format, and generate contrastive summaries (Echo Chambers). This requires high throughput, strict JSON adherence, and low latency.

## Alternatives Considered
1. **OpenAI GPT-4o / GPT-4o-mini:** Proprietary, closed-source models accessed via paid API.
2. **Meta Llama-3-8B-Instruct:** Open-weights model, hosted via HuggingFace Serverless Inference Endpoints.

## Decision
We selected **Meta Llama-3-8B-Instruct**.

## Justification
1. **Cost at Scale:** Running claim extraction on 50+ articles per search query results in massive token throughput. Relying on GPT-4o would incur unsustainable API costs. Llama 3 8B hosted on HuggingFace is significantly more cost-effective (and often free under certain tiers) for high-volume NLP pipelines.
2. **Task Specificity:** Llama-3-8B-Instruct is highly capable of following strict formatting instructions (like outputting raw JSON lists) when prompted correctly. We do not need the deep reasoning capabilities of GPT-4 to identify sentences in a text; a fast, smaller model like 8B is perfectly suited for extraction and summarization.
3. **Latency:** Llama-3-8B is much smaller than frontier models, meaning inference times (Time-To-First-Token) are incredibly fast. When orchestrated asynchronously across a distributed actor model, it allows the entire pipeline to finish in seconds rather than minutes.
