# Interview Prep: Tuning Cosine Similarity Thresholds

**Question:** *"Walk me through a time you had to fine-tune a machine learning parameter. How did you balance false positives and false negatives?"*

### The Context
In BiasScope, we group individual news claims into "Events" using Semantic Clustering. We generate a 384-dimensional vector embedding for every claim using `sentence-transformers`, and then calculate the Cosine Similarity between them using PostgreSQL `pgvector`.

If the similarity score is above a certain threshold, we merge them into the same Event.

### The Problem
When I initially built the clustering engine, I set the Cosine Similarity threshold to **0.75**. 

However, when I reviewed the database, I found massive "False Positives" (Under-clustering).
- Claim 1: "The President signed the tax bill."
- Claim 2: "The President vetoed the infrastructure bill."

Because both claims had the same subject ("The President") and similar actions regarding a "bill", the embedding model placed them close together in the vector space (Similarity: 0.78). My system merged them into the exact same event. This ruined our Echo Chamber summaries, because the LLM was getting completely contradictory facts grouped together.

So, I bumped the threshold up to **0.95**.
This created the opposite problem: "False Negatives" (Over-segmentation).
- Claim 1: "The President signed the tax bill today."
- Claim 2: "Today, the tax legislation was signed by the President."

These mean the exact same thing, but because the sentence structure was different, their similarity score was 0.88. At a 0.95 threshold, the system created two separate Events. Our consensus algorithm broke because it couldn't see that the Left and Right were reporting on the same thing.

### The Solution
I had to find the mathematical "Goldilocks Zone". I built a small ground-truth dataset of 100 claims and manually labeled which ones should be grouped together.

I ran a script to evaluate Precision (avoiding false positives) and Recall (avoiding false negatives) across thresholds from 0.70 to 0.99.

I discovered that a threshold of **0.85** provided the optimal F1 score. At 0.85, the model correctly grouped structural variations of the same claim, but strictly separated claims where the verb or entity completely changed the factual outcome. This stabilized the Event generation and drastically improved the quality of the LLM-generated Echo Chamber summaries.
