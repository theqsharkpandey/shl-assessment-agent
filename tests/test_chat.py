from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Ensure the API health route is responsive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_schema_validation():
    """Ensure invalid payloads are rejected by Pydantic."""
    # Missing 'messages' key
    response = client.post("/chat", json={"invalid": "payload"})
    assert response.status_code == 422

    # Messages array is empty (violates min_length=1)
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == 422

    # Missing role
    response = client.post("/chat", json={"messages": [{"content": "hello"}]})
    assert response.status_code == 422
