"""
Shared Prisma client singleton.

Pulled out of app/main.py so that other modules (auth dependencies, Celery
tasks, one-off scripts) can get a database handle without importing the
FastAPI app module itself — importing app.main has side effects (app
construction, CORS middleware registration, route registration) that have
nothing to do with wanting a DB connection.
"""

from .prisma_client import Prisma

prisma = Prisma()
