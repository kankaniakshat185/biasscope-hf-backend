# Deep Dive: `app/services/validation.py`

This file is responsible for mathematically evaluating the quality and bias of the dataset we have collected for a search query. It calculates two critical metrics: the **Polarization Score** and the **Data Quality Score (DQS)**.

---

## 1. Polarization Score (Jensen-Shannon Divergence)

The Polarization Score answers the question: *"Are the Left and the Right talking about this topic in completely different ways, or do they agree?"*

To measure this, we look at the **Sentiment Distributions**. 
- Does the Left view this event as 90% Positive?
- Does the Right view this event as 10% Positive and 90% Negative?

If their distributions are very different, the topic is Highly Polarized.

### The Math: Kullback-Leibler (KL) Divergence
To compare two probability distributions ($P$ and $Q$), mathematicians use **KL Divergence**. It measures how much information is lost if you use $Q$ to approximate $P$.

$$ D_{KL}(P || Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right) $$

**The Problem with KL Divergence:** It is not symmetric! $D_{KL}(P || Q)$ is not equal to $D_{KL}(Q || P)$. You can't just say "The distance between Left and Right is X." 

### The Solution: Jensen-Shannon (JS) Divergence
JS Divergence fixes this by creating a "middle ground" distribution $M$, which is exactly halfway between $P$ (Left) and $Q$ (Right):
$$ M = \frac{1}{2}(P + Q) $$

Then, it measures the KL Divergence from Left to Middle, and from Right to Middle, and averages them:

$$ JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M) $$

### In our Code:
```python
def jensen_shannon_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
```
- `p` is the sentiment distribution of the Left (e.g., `[Positive%, Neutral%, Negative%]`).
- `q` is the sentiment distribution of the Right.
- The result is a number between `0.0` (Perfectly Unified) and `1.0` (Perfectly Polarized). We multiply by 100 to get a score out of 100.

---

## 2. Data Quality Score (DQS)

The Data Quality Score evaluates whether the search results we gathered are actually trustworthy and comprehensive. It is a weighted formula out of 100.

### Component 1: Volume Score (25%)
You can't trust a bias analysis if you only analyzed 2 articles. You need volume.
- Formula: $\min\left(\frac{\text{Valid Articles}}{20}, 1.0\right)$
- If we have 20 or more valid articles, we get 100% for this component.

### Component 2: Diversity Score (25%)
Are all the articles coming from the exact same publisher? That's an echo chamber, not a balanced dataset.
- Formula: $\min\left(\frac{\text{Unique Sources}}{\text{Target Sources}}, 1.0\right)$
- We expect a healthy dataset to have at least a few unique sources (e.g., target = 5).

### Component 3: Yield Score (25%)
Did our scraper fail to read half the websites? 
- Formula: $\frac{\text{Valid Articles}}{\text{Total Attempted Articles}}$
- If we scraped 10 articles but 5 were blocked by paywalls or had invalid text, our yield is 50%.

### Component 4: Content Richness (25%)
Are the articles long, detailed pieces of journalism, or just 3-sentence stubs?
- Formula: $\min\left(\frac{\text{Average Characters per Article}}{3000}, 1.0\right)$
- We benchmark against 3000 characters as a healthy, full-length news article.

### The Final Calculation:
$$ \text{DQS} = (Volume \times 25) + (Diversity \times 25) + (Yield \times 25) + (Richness \times 25) $$

If the final DQS is below 50, we usually flag the dataset as potentially unreliable for deep macro-analysis.
