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

import hashlib
import logging
import os

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# Default model for pipeline stages that don't specify one (extraction,
# canonicalization, event summaries).
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Narrative/chat-style stages historically called HuggingFace directly with
# a different model, bypassing this cache entirely. cached_llm_call() now
# takes a `model` argument instead of hardcoding one, so every call site —
# including those — can go through here and still show up in
# /debug/llm-usage and /debug/cache-stats.
#
# Was "meta-llama/Meta-Llama-3-8B-Instruct" — as of 2026-08 that model is no
# longer served by any provider enabled on the HF Inference Router, so every
# narrative/chat call failed with a 400 ("not supported by any provider you
# have enabled") and silently fell back. Reusing MODEL_ID keeps narrative/
# chat on a model confirmed working (it's what the extraction stage already
# calls successfully).
NARRATIVE_MODEL_ID = MODEL_ID

_clients: dict = {}

def _get_client(model: str):
    if model not in _clients:
        token = os.getenv("HF_TOKEN")
        if not token:
            return None
        _clients[model] = InferenceClient(model=model, token=token)
    return _clients[model]

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
    model: str = MODEL_ID,
    temperature: float = 0.1,
) -> str:
    """
    Call the LLM with prompt-level caching.

    Args:
        prisma: Prisma client instance
        stage: Pipeline stage name (extraction, canonicalization, narrative, ...)
        system_prompt: System message
        user_prompt: User message
        max_tokens: Maximum response tokens
        model: HuggingFace model id. Defaults to the extraction-pipeline
            model; narrative/chat call sites pass NARRATIVE_MODEL_ID.
        temperature: Sampling temperature — narrative/chat call sites tend
            to want more variation than the low-temperature extraction
            stages, so this is no longer hardcoded to 0.1 for everyone.

    Returns:
        Cleaned LLM response string
    """
    prompt_hash = _compute_prompt_hash(model, system_prompt, user_prompt)

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
                    "model": model,
                    "cached": True,
                    "promptTokens": 0,
                    "completionTokens": 0,
                }
            )
            logger.info(f"[CACHE HIT] stage={stage} model={model} hash={prompt_hash[:12]}...")
            return cached.response
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    # ── Call LLM ──
    client = _get_client(model)
    if not client:
        logger.warning("No HF_TOKEN — cannot call LLM.")
        return ""

    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
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
                    "model": model,
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
                    "model": model,
                    "cached": False,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                }
            )
        except Exception:
            pass

        logger.info(f"[LLM CALL] stage={stage} model={model} ~{prompt_tokens}+{completion_tokens} tokens")
        return content

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


