# Deep Dive: `app/services/clustering.py`

This file is responsible for **Semantic Claim Clustering**. It takes individual sentences (claims) extracted from news articles and figures out if they are talking about the exact same real-world event.

If you have 100 articles about an election, they might all say "The candidate won" in 100 different ways. This file groups those 100 claims into **1 Canonical Claim** (an Event).

---

## 1. The Theory: What are Embeddings? (A for Apple)

Imagine you want to sort fruits. You could measure them on two axes:
- **X-axis:** How sweet is it? (0 to 10)
- **Y-axis:** How crunchy is it? (0 to 10)

An Apple might be `[7, 8]`. A Banana might be `[9, 1]`. 

If you plot them on a graph, fruits that are similar will be physically close to each other. 

In NLP (Natural Language Processing), we do this with words and sentences, but instead of just 2 axes (sweetness and crunchiness), we use **384 axes (dimensions)**. 
An **Embedding** is just a list of 384 numbers that represents the *meaning* of a sentence.

If two sentences have similar meanings, their 384-number lists will look very similar, and they will be plotted close to each other in this 384-dimensional space.

---

## 2. The Math: Cosine Similarity

Once we have our sentences turned into coordinates (vectors), how do we measure the distance between them?

We use **Cosine Similarity**. Instead of measuring the straight-line distance (Euclidean distance) between two points, we measure the **Angle ($\theta$)** between the arrows pointing to them from the origin $(0,0)$.

Why? Because if one sentence is long and the other is short, their points might be far apart in straight-line distance, but they are pointing in the exact same *semantic direction*.

### The Formula

Given two vectors (lists of numbers) $A$ and $B$:

$$ \text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{||A|| ||B||} $$

Let's break that down:
1. **$A \cdot B$ (Dot Product):** You multiply each corresponding number in the lists and add them up. $(A_1 \times B_1) + (A_2 \times B_2) ...$
2. **$||A||$ (Magnitude):** The actual length of the arrow. Found by squaring all numbers in $A$, adding them up, and taking the square root.
3. You divide the Dot Product by the Magnitudes multiplied together.

The result is a number between **-1** and **1**:
- **1.0**: The angle is 0°. The sentences mean the EXACT same thing.
- **0.0**: The angle is 90°. The sentences have nothing to do with each other.
- **-1.0**: The angle is 180°. The sentences mean the exact opposite.

---

## 3. The Code: How it works in `clustering.py`

### Step 1: Loading the Model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```
We use `all-MiniLM-L6-v2`. It's a small, incredibly fast model created by HuggingFace that takes a sentence and spits out a 384-dimensional vector. 

### Step 2: Generating Embeddings
```python
embeddings = model.encode([claim_text])
vector = embeddings[0].tolist()
```
When a new claim comes in, we pass the text to the model. It returns our list of 384 numbers (the embedding). We convert it to a standard Python list.

### Step 3: Searching the Database (`pgvector`)
```python
# Prisma raw query using pgvector
results = await prisma.query_raw(f'''
    SELECT id, text, "canonicalId", 1 - (embedding <=> '{vector_str}') AS similarity
    FROM "Claim"
    WHERE 1 - (embedding <=> '{vector_str}') > 0.85
    ORDER BY similarity DESC
    LIMIT 1;
''')
```
This is where the magic happens. We don't want to pull every single claim from the database into Python to calculate the Cosine Similarity—that would be incredibly slow.

Instead, we use a PostgreSQL extension called **`pgvector`**. 
In `pgvector`, the operator `<=>` calculates the **Cosine Distance**. 
- Cosine Distance = $1 - \text{Cosine Similarity}$.

So, `1 - (embedding <=> vector)` reverses it back into **Cosine Similarity**.
We ask the database: *"Find me claims where the similarity is greater than 0.85 (85% similar), and give me the best match."*

### Step 4: Clustering
- **If a match > 0.85 is found:** We link this new claim to the existing `canonicalId` (Event). We are saying, "This is just a new article reporting on an event we already know about."
- **If NO match is found:** We create a brand new `Event` in the database, because this is a story we haven't seen before.
