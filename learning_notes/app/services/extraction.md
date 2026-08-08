# Deep Dive: `app/services/extraction.py`

This file is the core of our **Claim-Centric** architecture. 

Historically, bias analysis tools analyzed the *entire* article and assigned it a single sentiment score. But articles are complex; they might report a factual, positive event in paragraph 1, and a subjective, negative opinion in paragraph 3. 

To solve this, BiasScope breaks articles down into **Claims**.

---

## 1. What is a Claim?

A claim is a single, atomic, falsifiable statement.
- **Article Text:** *"The President signed the controversial tax bill today, despite fierce opposition from the opposing party who claim it will ruin the economy."*
- **Extracted Claim 1:** *"The President signed a tax bill today."*
- **Extracted Claim 2:** *"The opposing party claims the tax bill will ruin the economy."*

By breaking text into claims, our downstream machine learning models (Clustering, Sentiment, Bias) can operate on highly specific, clean data, vastly reducing noise and hallucinations.

---

## 2. The Extraction Prompt (Llama 3)

We use an LLM (`Meta-Llama-3-8B-Instruct`) to read the raw article text and extract these claims. 

### The Prompt Design
The prompt is meticulously engineered to enforce strict structural outputs (JSON) and prevent the LLM from hallucinating or inserting its own knowledge.

```text
You are a strict, objective factual extraction system. 
Your ONLY job is to extract discrete, factual claims made within the provided text.
DO NOT use outside knowledge. DO NOT hallucinate.
Return the result strictly as a JSON list of strings.
```

### Why JSON?
By forcing the LLM to output valid JSON (a list of strings `["Claim 1", "Claim 2"]`), our Python code can instantly parse the output using `json.loads()` and iterate through the claims. If the LLM outputs conversational text ("Here are the claims you requested..."), it breaks our automated pipeline.

---

## 3. The Code: `process_and_store_claims`

When an article is passed into this function, the following happens:
1. **Extraction:** We call `llm_client.py` to ask Llama 3 to extract the claims from the article.
2. **Parsing:** We safely parse the JSON output. If the LLM hallucinates or formats it poorly, we catch the exception and fallback gracefully.
3. **Storage:** Each individual claim string is saved to the database (`prisma.claim.create`) and linked back to the parent `Article`.

This means our database has a 1-to-Many relationship:
`1 Article` $\rightarrow$ `Many Claims`

From this point forward, the rest of the application (Clustering, Echo Chambers) completely ignores the original Article text and only operates on these isolated Claims!
