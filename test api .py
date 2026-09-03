"""
test_api.py — basic smoke tests for the FastAPI app.

These are the tests the CI/CD pipeline runs on every push. They check
that the app starts correctly and the health endpoint responds — they
do NOT call the real Gemini API (no API key needed to run these).
"""

from fastapi.testclient import TestClient
from main import app


def test_health_check():
    """The /health endpoint should respond even without GEMINI_API_KEY set."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_ask_without_api_key_returns_503():
    """Without GEMINI_API_KEY, /ask should fail gracefully with a clear error,
    not crash the server."""
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "test question"})
        # Either 503 (no key set) or 200 (key set in this environment) is acceptable —
        # the key assertion is that the server does not crash (no 500 error).
        assert response.status_code in (200, 503)
