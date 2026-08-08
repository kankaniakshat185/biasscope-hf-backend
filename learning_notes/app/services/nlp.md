# Deep Dive: `app/services/nlp.py`

This file is the **Intelligence Core** of the application. Once claims are extracted from articles, this file runs them through machine learning models to figure out two main things:
1. What is the sentiment of the text? (Positive, Negative, Neutral)
2. What is the political bias of the text? (Left, Center, Right)

---

## 1. The Theory: HuggingFace Pipelines

In machine learning, you usually have to do a lot of complex work to take raw text, turn it into numbers (tokenization), run it through a massive neural network (inference), and turn the numbers back into a human-readable label (post-processing).

HuggingFace's `pipeline` abstracts all of that into one line of code.

### Sentiment Analysis
We use `cardiffnlp/twitter-roberta-base-sentiment-latest`. 
- **Why this model?** Most sentiment models are trained on movie reviews or product reviews (e.g., "This movie was great!"). But news and social media are different. This model was trained on Twitter data, making it highly effective at understanding sarcasm, political rhetoric, and news language.
- It returns 3 classes: Positive, Negative, Neutral.

### Political Bias
We use `bucketresearch/politicalBiasBERT`.
- This is a BERT model fine-tuned on political text to detect whether a sentence is framed from a Left, Center, or Right-leaning perspective.

---

## 2. Echo Chambers & Contrastive Summaries

One of the coolest features in this file is the **Echo Chamber Analysis** (`generate_contrastive_summaries`).

### How it works:
When we get a bunch of claims about an event, we group them by the political leaning of the publisher.
- `left_claims`: Claims published by Left-leaning sources (CNN, MSNBC, etc.)
- `right_claims`: Claims published by Right-leaning sources (Fox News, Breitbart, etc.)

We then use an LLM (Llama 3) to summarize *only* the left claims, and then summarize *only* the right claims. 
This produces two distinct narratives of the exact same event, revealing how different media ecosystems are spinning the story.

---

## 3. The Math: Sentiment Scores

When the sentiment model predicts a label, it also gives a **Probability (Confidence Score)** between 0.0 and 1.0.

For example:
- Positive: 0.80
- Neutral: 0.15
- Negative: 0.05

Instead of just saying "This is Positive", we convert these probabilities into a **Continuous Sentiment Score** from -1.0 to 1.0.

### The Formula:
$$ \text{Sentiment Score} = (\text{Positive Probability} \times 1.0) + (\text{Negative Probability} \times -1.0) + (\text{Neutral Probability} \times 0.0) $$

In our example:
$$ (0.80 \times 1.0) + (0.05 \times -1.0) + (0.15 \times 0.0) = 0.80 - 0.05 = \mathbf{0.75} $$

This means the text is strongly positive (+0.75). If it were mostly negative, the score would be negative (e.g., -0.60).

---

## 4. Entity Extraction

The function `extract_entity_sentiment` looks at the claims to find the "Entities" being talked about (People, Organizations, Countries).

It uses **Spacy** (`spacy.load("en_core_web_trf")`), which is a powerful NLP library. 
Spacy performs **Named Entity Recognition (NER)**. It reads a sentence like:
*"Joe Biden visited the White House today."*
And it tags:
- `Joe Biden` $\rightarrow$ PERSON
- `White House` $\rightarrow$ FAC (Facility/Building)

We then calculate the average sentiment of all sentences that mention a specific entity to figure out if the media is talking positively or negatively about that person or thing!
