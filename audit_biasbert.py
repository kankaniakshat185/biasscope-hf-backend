from transformers import pipeline

print("Loading PoliticalBiasBERT...")
bias_pipeline = pipeline("text-classification", model="bucketresearch/politicalBiasBERT", truncation=True, max_length=512)

tests = [
    {"source": "Fox News", "text": "The radical left continues to push their socialist agenda, ignoring the working class and destroying the economy with massive tax hikes and unnecessary regulations."},
    {"source": "CNN", "text": "Progressive lawmakers are championing a bold new initiative to combat systemic inequality and ensure a fair, living wage for marginalized communities across the nation."},
    {"source": "Reuters", "text": "The central bank announced a 0.25% interest rate increase on Wednesday, aiming to curb inflation while maintaining steady job growth in the manufacturing sector."},
    {"source": "Gizmodo", "text": "The billionaire class is hoarding wealth while the planet burns, and frankly, the latest tech bro vanity project is an absolute disgrace to humanity."},
    {"source": "BBC", "text": "Parliament voted 312 to 298 in favor of the new trade agreement, concluding months of contentious negotiations between the ruling party and the opposition."}
]

print("\n--- PoliticalBiasBERT Audit ---")
for t in tests:
    res = bias_pipeline(t["text"])[0]
    print(f"\nSource: {t['source']}")
    print(f"Text Snippet: {t['text'][:60]}...")
    print(f"Predicted Bias: {res['label']} (Confidence: {res['score']:.4f})")

