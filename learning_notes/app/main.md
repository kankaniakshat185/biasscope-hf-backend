# Deep Dive: `app/main.py`

This file is the **FastAPI Application Entry Point**. It is the bridge between the internet and our NLP engine.

---

## 1. FastAPI Fundamentals

FastAPI is a modern, high-performance web framework for Python. 
We use it because it is natively asynchronous (`async` / `await`), which is critical for I/O bound tasks like making HTTP requests to HuggingFace or querying a PostgreSQL database.

### The App Instance
```python
app = FastAPI(title="Biascope API")
```
This initializes the server.

### CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], ...
)
```
**CORS (Cross-Origin Resource Sharing)** is a security feature built into web browsers. If our frontend is hosted on `vercel.app` and our backend is on `huggingface.co`, the browser will block the request unless the backend explicitly says, "I allow requests from anywhere (`*`)."

---

## 2. The Prisma ORM

```python
prisma = Prisma()

@app.on_event("startup")
async def startup():
    await prisma.connect()
```
We use **Prisma** as our ORM (Object-Relational Mapper).
Instead of writing raw SQL strings (`SELECT * FROM users`), an ORM lets us write Python code (`await prisma.user.find_many()`), and it translates it to secure SQL automatically. 

We connect to the database when the FastAPI server starts up, and disconnect when it shuts down.

---

## 3. The Endpoints

### `POST /search`
This is the primary endpoint hit by the frontend when a user searches for a topic (e.g., "AI Regulation").

**The Asynchronous Hand-off:**
If a user searches for a topic, scraping 50 articles, running Llama-3 extraction, computing embeddings, and clustering could take 30 to 60 seconds. 

If we made the user wait on the HTTP request for 60 seconds, their browser would probably time out, and our server would block other users from making requests.

Instead, `/search` does this:
1. Instantly creates a "Search" record in the database with status `PENDING`.
2. Hands the heavy lifting off to a background worker (Celery) using `snapshot_task.delay(search_id, query)`.
3. Instantly returns the `search_id` to the frontend.

The frontend then uses a different endpoint (`GET /search/{id}`) to poll the database every 2 seconds until the status changes from `PENDING` to `COMPLETED`.
