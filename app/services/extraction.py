import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Initialize the embedding model globally so it's only loaded once.
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def embed_text(text: str) -> List[float]:
    """Generates a 384-dimensional embedding for the given text."""
    model = get_embedding_model()
    # model.encode returns a numpy array, we convert to list
    return model.encode(text).tolist()

def get_topic_relevance(query: str, article_text: str, title: str) -> float:
    """Calculates a relevance score based on headline presence, first paragraph presence, and frequency."""
    if not query:
        return 1.0
        
    query_lower = query.lower()
    text_lower = article_text.lower()
    title_lower = (title or "").lower()
    
    score = 0.0
    if query_lower in title_lower:
        score += 0.5
        
    first_paragraph = text_lower[:500]
    if query_lower in first_paragraph:
        score += 0.3
        
    freq = text_lower.count(query_lower)
    if freq >= 3:
        score += 0.2
    elif freq > 0:
        score += 0.1
        
    return min(score, 1.0)

def extract_claims(article_text: str) -> List[Dict[str, Any]]:
    """
    Extracts factual, objective claims from the article using Qwen 2.5 or Llama 3.
    Returns a list of dicts: [{"text": "...", "confidence": 0.95}]
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found. Skipping claim extraction.")
        return []
        
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model=model_id, token=hf_token)
    
    system_prompt = (
        "You are a strict factual extraction engine. Your ONLY purpose is to extract objective, verifiable events and actions from the text.\n"
        "RULES:\n"
        "1. Extract ONLY factual assertions (e.g., 'SpaceX signed a compute agreement with Anthropic.').\n"
        "2. DO NOT extract opinions, sentiment, speculation, or editorial framing.\n"
        "3. Claims must be self-contained and highly specific. Do not use pronouns like 'He' or 'They' if the entity is known.\n"
        "4. Assign a confidence score (0.0 to 1.0) indicating how explicitly the article states this fact.\n"
        "5. Respond ONLY with valid JSON. Do not include markdown code blocks (```json) or conversational text.\n"
        "Format:\n"
        "{\"claims\": [{\"text\": \"...\", \"confidence\": 0.95}]}"
    )
    
    user_prompt = f"Extract the claims from the following article text (limited to 4000 chars for context):\n\n{article_text[:4000]}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.1)
        content = response.choices[0].message.content.strip()
        
        # Clean up possible markdown code blocks
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        data = json.loads(content)
        return data.get("claims", [])
    except Exception as e:
        logger.error(f"Failed to extract claims: {e}")
        return []

async def process_and_canonicalize_claims(prisma, article_id: str, article_text: str, source: str, url: str, published_at, query: str = "", title: str = ""):
    """
    Executes Phase 2 Pipeline:
    1. Topic Relevance Scoring (Skip if < 0.6)
    2. Extract Claims
    3. Embed
    4. Canonicalize (Merge if similarity > 0.92)
    5. Link Evidence
    """
    relevance_score = get_topic_relevance(query, article_text, title)
    if relevance_score < 0.6:
        logger.info(f"Article has low relevance score: {relevance_score}. Skipping claim extraction.")
        return []

    raw_claims = extract_claims(article_text)
    if not raw_claims:
        return []
        
    processed_claims = []
    
    for c in raw_claims:
        text = c.get("text", "").strip()
        confidence = c.get("confidence", 0.8)
        
        if not text or len(text) < 15:
            continue
            
        # 1. Generate Embedding
        embedding_list = embed_text(text)
        
        # Convert list to pgvector string format for raw SQL: '[0.1, 0.2, ...]'
        vector_str = "[" + ",".join(map(str, embedding_list)) + "]"
        
        # 2. Canonicalization - Query for existing claims using Cosine Distance (<=>)
        # Cosine similarity = 1 - Cosine distance. A similarity of > 0.92 means distance < 0.08
        similarity_threshold = 0.92
        distance_threshold = 1.0 - similarity_threshold
        
        # We must use raw SQL because Prisma doesn't natively support pgvector distance operators yet
        matched_claims = await prisma.query_raw(
            f'''
            SELECT id, "canonicalClaim", "confidence", (embedding <=> '{vector_str}'::vector) as distance 
            FROM "claim" 
            WHERE (embedding <=> '{vector_str}'::vector) < {distance_threshold}
            ORDER BY distance ASC 
            LIMIT 1
            '''
        )
        
        canonical_claim_id = None
        
        if matched_claims and len(matched_claims) > 0:
            # 3A. Merge with Existing Canonical Claim
            match = matched_claims[0]
            canonical_claim_id = match["id"]
            
            # Update moving average of confidence
            # A simple approach: avg of old and new confidence. 
            old_conf = float(match["confidence"])
            new_conf = (old_conf + float(confidence)) / 2.0
            
            await prisma.claim.update(
                where={"id": canonical_claim_id},
                data={"confidence": new_conf}
            )
        else:
            # 3B. Create New Canonical Claim
            new_claim = await prisma.query_raw(
                f'''
                INSERT INTO "claim" ("id", "canonicalClaim", "confidence", "embedding", "createdAt") 
                VALUES (gen_random_uuid()::text, '{text.replace("'", "''")}', {confidence}, '{vector_str}'::vector, NOW())
                RETURNING id;
                '''
            )
            if new_claim:
                canonical_claim_id = new_claim[0]["id"]
            else:
                continue # Skip if insert failed
        
        # 4. Link Evidence
        if canonical_claim_id:
            await prisma.evidence.create(
                data={
                    "claimId": canonical_claim_id,
                    "articleId": article_id,
                    "sentence": text,
                    "source": source,
                    "url": url,
                    "publishedAt": published_at or datetime.now(),
                    "stance": "MENTION" # Default for extraction phase
                }
            )
            processed_claims.append(canonical_claim_id)
            
    return processed_claims
