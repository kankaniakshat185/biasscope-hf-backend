import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def embed_text(text: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()

def call_extraction_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return ""
    # FIX 10: Replace Llama-3 with Qwen2.5
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    client = InferenceClient(model=model_id, token=hf_token)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return content
    except Exception as e:
        logger.error(f"Extraction LLM call failed: {e}")
        return ""

def get_topic_relevance(query: str, article_text: str, title: str) -> float:
    if not query: return 1.0
    query_lower = query.lower()
    text_lower = article_text.lower()
    title_lower = (title or "").lower()
    score = 0.0
    if query_lower in title_lower: score += 0.5
    if query_lower in text_lower[:500]: score += 0.3
    freq = text_lower.count(query_lower)
    if freq >= 3: score += 0.2
    elif freq > 0: score += 0.1
    return min(score, 1.0)

def get_claim_relevance(query: str, claim_text: str, article_relevance: float) -> float:
    # FIX 1: Claim-level relevance
    if not query: return 1.0
    query_lower = query.lower()
    claim_lower = claim_text.lower()
    score = article_relevance * 0.5
    if query_lower in claim_lower:
        score += 0.5
    return min(score, 1.0)

def extract_claims(article_text: str) -> List[Dict[str, Any]]:
    # FIX 2: Store real evidence in "evidence_sentence"
    system_prompt = (
        "You are a strict factual extraction engine. Your ONLY purpose is to extract objective, verifiable events and actions from the text.\n"
        "RULES:\n"
        "1. Extract ONLY factual assertions (e.g., 'SpaceX signed a compute agreement with Anthropic.').\n"
        "2. DO NOT extract opinions, sentiment, speculation, or editorial framing.\n"
        "3. Claims must be self-contained and highly specific. Do not use pronouns like 'He' or 'They' if the entity is known.\n"
        "4. Assign a confidence score (0.0 to 1.0) indicating how explicitly the article states this fact.\n"
        "5. Provide the exact 'evidence_sentence' from the text that proves this claim.\n"
        "6. Respond ONLY with valid JSON.\n"
        "Format:\n"
        "{\"claims\": [{\"text\": \"...\", \"confidence\": 0.95, \"evidence_sentence\": \"...\"}]}"
    )
    user_prompt = f"Extract the claims from the following article text (limited to 4000 chars for context):\n\n{article_text[:4000]}"
    
    resp = call_extraction_llm(system_prompt, user_prompt)
    if resp:
        try:
            return json.loads(resp).get("claims", [])
        except json.JSONDecodeError:
            pass
    return []

def generate_canonical_claim(existing_claims: List[str], new_claim: str) -> str:
    # FIX 4: LLM Canonicalization
    system_prompt = (
        "You are an AI tasked with canonicalizing related claims. "
        "Review the existing claims and the new claim. "
        "Combine them into a single, comprehensive canonical claim that captures the complete fact accurately without redundancy. "
        "Return ONLY valid JSON matching this schema: "
        '{"canonical_claim": "..."}'
    )
    payload = {
        "existing_claims": existing_claims,
        "new_claim": new_claim
    }
    user_prompt = json.dumps(payload)
    resp = call_extraction_llm(system_prompt, user_prompt, max_tokens=256)
    if resp:
        try:
            return json.loads(resp).get("canonical_claim", new_claim)
        except json.JSONDecodeError:
            pass
    return new_claim

async def process_and_canonicalize_claims(prisma, article_id: str, article_text: str, source: str, url: str, published_at, query: str = "", title: str = ""):
    article_relevance = get_topic_relevance(query, article_text, title)
    
    # FIX 1: Don't discard entire article, extract first
    raw_claims = extract_claims(article_text)
    if not raw_claims:
        return []
        
    processed_claims = []
    
    for c in raw_claims:
        text = c.get("text", "").strip()
        confidence = c.get("confidence", 0.8)
        evidence_sentence = c.get("evidence_sentence", text).strip()
        
        if not text or len(text) < 15:
            continue
            
        claim_relevance = get_claim_relevance(query, text, article_relevance)
        if claim_relevance < 0.6:
            continue # FIX 1: Discard only low relevance claims
            
        embedding_list = embed_text(text)
        vector_str = "[" + ",".join(map(str, embedding_list)) + "]"
        
        # FIX 5: Lower similarity threshold to 0.82
        similarity_threshold = 0.82
        distance_threshold = 1.0 - similarity_threshold
        
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
            match = matched_claims[0]
            canonical_claim_id = match["id"]
            
            # FIX 4: Real Canonicalization
            old_canonical = match["canonicalClaim"]
            new_canonical = generate_canonical_claim([old_canonical], text)
            
            old_conf = float(match["confidence"])
            new_conf = (old_conf + float(confidence)) / 2.0
            
            # FIX 3: Parameterized queries where possible (using ORM here)
            await prisma.claim.update(
                where={"id": canonical_claim_id},
                data={
                    "canonicalClaim": new_canonical,
                    "confidence": new_conf
                }
            )
        else:
            # FIX 3: Removed raw SQL interpolation for text/confidence
            new_claim = await prisma.query_raw(
                f'''
                INSERT INTO "claim" ("id", "canonicalClaim", "confidence", "embedding", "createdAt") 
                VALUES (gen_random_uuid()::text, $1, $2, '{vector_str}'::vector, NOW())
                RETURNING id;
                ''',
                text, float(confidence)
            )
            if new_claim:
                canonical_claim_id = new_claim[0]["id"]
            else:
                continue
        
        if canonical_claim_id:
            await prisma.evidence.create(
                data={
                    "claimId": canonical_claim_id,
                    "articleId": article_id,
                    "sentence": evidence_sentence, # FIX 2: Store real evidence sentence
                    "source": source,
                    "url": url,
                    "publishedAt": published_at or datetime.now(),
                    "stance": "MENTION"
                }
            )
            processed_claims.append(canonical_claim_id)
            
    return processed_claims
