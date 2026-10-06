from io import BytesIO
from uuid import uuid4
from unittest.mock import patch

from smartkcet.main import app


def test_admin_upload_preview_only_returns_questions():
    sample_text = """Q1. Which of the following is a scalar quantity?
A. Force
B. Velocity
C. Mass
D. Acceleration
Answer: C

Q2. A body of mass 2 kg is moving with a speed of 3 m/s. Its kinetic energy is:
A. 6 J
B. 9 J
C. 12 J
D. 18 J
Answer: B
"""
    filename = f"preview-{uuid4().hex[:8]}.txt"

    with app.test_client() as client:
        with patch("smartkcet.admin.upload.require_admin", return_value={"role": "admin", "sub": "admin@vyasaprep.com"}):
            response = client.post(
                "/api/admin/upload",
                data={
                    "subject": "Physics",
                    "preview_only": "true",
                    "file": (BytesIO(sample_text.encode("utf-8")), filename),
                },
            )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert "preview_questions" in payload
    assert len(payload["preview_questions"]) >= 1
    assert payload["preview_questions"][0]["q"]
