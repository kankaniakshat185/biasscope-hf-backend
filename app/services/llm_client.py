"""
Centralized LLM Client — Features 1 & 7

All LLM calls go through this module.
Provides:
  - Prompt-level SHA-256 caching (Feature 1)
  - Per-stage usage analytics (Feature 7)
  - Single InferenceClient instance
  - JSON cleanup for all responses

Usage:
    from app.services.llm_client import cached_llm_call
    result = await cached_llm_call(prisma, "extraction", system_prompt, user_prompt)
"""

import os
import json
import hashlib
import logging
from typing import Optional
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_client = None

def _get_client():
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            return None
        _client = InferenceClient(model=MODEL_ID, token=token)
    return _client

def _compute_prompt_hash(model: str, system_prompt: str, user_prompt: str) -> str:
    raw = f"{model}|{system_prompt}|{user_prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _clean_llm_response(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return content


async def cached_llm_call(
    prisma,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """
    Call the LLM with prompt-level caching.
    
    Args:
        prisma: Prisma client instance
        stage: Pipeline stage name (extraction, canonicalization, merge, summary)
        system_prompt: System message
        user_prompt: User message
        max_tokens: Maximum response tokens
    
    Returns:
        Cleaned LLM response string
    """
    prompt_hash = _compute_prompt_hash(MODEL_ID, system_prompt, user_prompt)

    # ── Check cache ──
    try:
        cached = await prisma.llmcache.find_unique(
            where={"promptHash": prompt_hash}
        )
        if cached:
            # Log cache hit
            await prisma.llmusage.create(
                data={
                    "stage": stage,
                    "model": MODEL_ID,
                    "cached": True,
                    "promptTokens": 0,
                    "completionTokens": 0,
                }
            )
            logger.info(f"[CACHE HIT] stage={stage} hash={prompt_hash[:12]}...")
            return cached.response
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    # ── Call LLM ──
    client = _get_client()
    if not client:
        logger.warning("No HF_TOKEN — cannot call LLM.")
        return ""

    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        content = _clean_llm_response(resp.choices[0].message.content)

        # Estimate tokens
        prompt_tokens = len(system_prompt.split()) + len(user_prompt.split())
        completion_tokens = len(content.split())

        # ── Store in cache ──
        try:
            await prisma.llmcache.create(
                data={
                    "promptHash": prompt_hash,
                    "model": MODEL_ID,
                    "stage": stage,
                    "response": content,
                }
            )
        except Exception as e:
            logger.warning(f"Cache store failed (likely duplicate): {e}")

        # ── Log usage ──
        try:
            await prisma.llmusage.create(
                data={
                    "stage": stage,
                    "model": MODEL_ID,
                    "cached": False,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                }
            )
        except Exception:
            pass

        logger.info(f"[LLM CALL] stage={stage} ~{prompt_tokens}+{completion_tokens} tokens")
        return content

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


