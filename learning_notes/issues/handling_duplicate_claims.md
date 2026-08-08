# Interview Prep: Handling Duplicate Claims & Syndicated Content

**Question:** *"Tell me about a time you had to deal with dirty or skewed data in a machine learning pipeline."*

### The Context
BiasScope calculates the cross-ideological consensus of an event by looking at the diversity of publishers reporting on a specific claim. If 10 Left-leaning publishers and 10 Right-leaning publishers report on "Candidate X won the debate", we consider that claim highly corroborated.

### The Problem
During early testing, I noticed that certain claims were being flagged as massive, widely-reported events with 50+ publishers attached to them. But when I looked closer, the text of the articles from all 50 publishers was identical. 

I realized we were falling victim to **Syndicated Wire Content** (like Reuters or Associated Press). AP publishes an article, and 50 local news stations auto-publish the exact same article on their domains. 

Because we process claims, our system was extracting the exact same claims 50 times, clustering them together, and assuming 50 different journalists had independently corroborated the story. This artificially inflated the importance and consensus of wire stories and skewed the mathematical Polarization calculations.

### The Solution
I needed to stop this data from ever reaching the LLM or the database. I implemented a **Fuzzy Matching Deduplication** layer in the `cleaning.py` service.

1. As we scrape raw articles, I run them through a Levenshtein-based similarity algorithm.
2. I compare every incoming article to a running list of unique articles.
3. If the textual similarity ratio exceeds 0.95 (95% identical), I classify it as a syndicated duplicate and completely drop it from the pipeline.
4. If it's below 0.95, it's added to the unique list.

This ensured that our downstream Clustering algorithm only saw distinct, uniquely authored journalism, which restored mathematical integrity to our consensus and polarization scores.
