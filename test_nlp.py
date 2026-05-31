import asyncio
from app.services.ingestion import scrape_single_url
import spacy
from collections import Counter

spacy_nlp = spacy.load("en_core_web_sm")

async def main():
    art = await scrape_single_url("https://www.aljazeera.com/news/liveblog/2026/5/30/iran-war-live-trump-due-to-make-final-determination-on-deal-with-tehran")
    text = art["content"]
    doc = spacy_nlp(text[:2000])
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            name = ent.text.strip().title()
            if len(name) > 2 and "\n" not in name:
                if name not in entities:
                    entities[name] = {"label": ent.label_, "count": 1}
                else:
                    entities[name]["count"] += 1
    sorted_ents = sorted(entities.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
    art["entities"] = {k: v["label"] for k, v in sorted_ents}
    
    # extract keywords
    entity_counter = Counter()
    for entity in art["entities"].keys():
        entity_counter[entity] += 1
        
    most_common = entity_counter.most_common(10)
    top_keywords = [{"word": word, "count": count} for word, count in most_common]
    print(top_keywords)

asyncio.run(main())
