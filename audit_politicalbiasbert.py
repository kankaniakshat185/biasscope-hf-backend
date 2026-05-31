import os
from transformers import pipeline

def run_audit():
    print("="*60)
    print("  POLITICAL BIAS BERT - ADVERSARIAL AUDIT")
    print("="*60)
    print("Loading model 'bucketresearch/politicalBiasBERT'...")
    
    try:
        bias_pipeline = pipeline("text-classification", model="bucketresearch/politicalBiasBERT", truncation=True, max_length=512)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure transformers and torch are installed in your environment.")
        return

    # Benchmark tests covering explicit framing vs objective reporting
    tests = [
        {
            "id": "1 (CNN - Left Framing)",
            "text": "Progressive lawmakers are championing a bold new initiative to combat systemic inequality and ensure a fair, living wage for marginalized communities across the nation.",
            "expected": "LEFT"
        },
        {
            "id": "2 (Fox - Right Framing)",
            "text": "The radical left continues to push their socialist agenda, ignoring the working class and destroying the economy with massive tax hikes and unnecessary regulations.",
            "expected": "RIGHT"
        },
        {
            "id": "3 (Reuters - Neutral Reporting)",
            "text": "The central bank announced a 0.25% interest rate increase on Wednesday, aiming to curb inflation while maintaining steady job growth in the manufacturing sector.",
            "expected": "CENTER"
        },
        {
            "id": "4 (Gizmodo - Left Outrage)",
            "text": "The billionaire class is hoarding wealth while the planet burns, and frankly, the latest tech bro vanity project is an absolute disgrace to humanity.",
            "expected": "LEFT"
        },
        {
            "id": "5 (Breitbart - Right Populist)",
            "text": "Globalist elites are attempting to strip away our fundamental constitutional rights while completely opening the borders to unchecked illegal immigration.",
            "expected": "RIGHT"
        },
        {
            "id": "6 (Adversarial - Right quoting Left)",
            "text": "In a shocking display of radicalism, Senator Smith proudly stated, 'We must combat systemic inequality and ensure a living wage for all marginalized communities'.",
            "expected": "RIGHT" # A human knows it's a right-wing critique, but what will BERT do?
        }
    ]

    correct = 0
    print("\nRunning test cases...")
    for t in tests:
        res = bias_pipeline(t["text"])[0]
        prediction = res['label']
        confidence = res['score']
        
        match = "✅" if prediction == t["expected"] else "❌"
        if prediction == t["expected"]:
            correct += 1
            
        print("-" * 60)
        print(f"Test {t['id']}")
        print(f"Text Snippet: {t['text'][:80]}...")
        print(f"Prediction:   {prediction} (Confidence: {confidence:.4f})")
        print(f"Expected:     {t['expected']} {match}")
        
    print("=" * 60)
    print(f"Audit Complete: {correct}/{len(tests)} Correct ({(correct/len(tests))*100:.1f}%)")
    
    if correct == len(tests):
        print("Confidence Assessment: HIGH")
    elif correct >= len(tests) - 1:
        print("Confidence Assessment: MEDIUM-HIGH (Failed adversarial edge cases)")
    else:
        print("Confidence Assessment: LOW (Model struggles with basic ideological framing)")
        
if __name__ == "__main__":
    run_audit()
