"""Chat-with-context endpoints. Both routes were near-identical copies that
each constructed their own uncached HuggingFace client (Q3/A4 from the
audit) — collapsed into one helper that goes through cached_llm_call, so
every LLM call in the app shows up in /debug/llm-usage."""

from fastapi import APIRouter, Body, HTTPException

from ..db import prisma
from ..services.llm_client import NARRATIVE_MODEL_ID, cached_llm_call

router = APIRouter(tags=["chat"])


async def _chat_with_context(stage: str, context: str, message: str) -> dict:
    system_prompt = (
        "You are an expert AI intelligence analyst. Use the following context to "
        "answer the user's question accurately. Do not invent information outside "
        "the context. If the user asks something unrelated to it, politely decline.\n\n"
        f"Context:\n{context[:2500]}"
    )
    answer = await cached_llm_call(
        prisma, stage, system_prompt, message,
        max_tokens=250, model=NARRATIVE_MODEL_ID, temperature=0.3,
    )
    if not answer:
        return {"answer": (
            "Couldn't reach the language model — check that HF_TOKEN is set and valid "
            "(a 403 usually means the HuggingFace account hasn't accepted the model's terms)."
        )}
    return {"answer": answer}


@router.post("/chat-with-article")
async def chat_with_article(articleId: str = Body(...), message: str = Body(...)):
    article = await prisma.article.find_unique(where={"id": articleId})
    if not article:
        raise HTTPException(status_code=404, detail="Article not found in database.")
    context = article.content if article.content else article.title
    return await _chat_with_context("chat_article", context, message)


@router.post("/chat-with-summary")
async def chat_with_summary(searchId: str = Body(...), message: str = Body(...)):
    insight = await prisma.insight.find_first(where={"searchId": searchId})
    if not insight:
        raise HTTPException(status_code=404, detail="Summary not found in database.")
    return await _chat_with_context("chat_summary", insight.narrativeSummary or "", message)
