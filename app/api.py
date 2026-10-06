"""Minimal API entry point (Role 3); search routes will be added later."""

from fastapi import FastAPI

app = FastAPI(title="Hybrid Document Search")


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "ok"}
