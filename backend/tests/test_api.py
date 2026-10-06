import concurrent.futures
import io
import pytest
from PIL import Image

def _create_minimal_pdf_bytes():
    """Generates a minimal valid PDF byte sequence starting with %PDF."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >> endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
        b"trailer << /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )

def _create_png_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color="white")
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health_check_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data

def test_similarity_rejects_empty_answer_key(test_client):
    response = test_client.post(
        "/similarity",
        data={"answer_key_text": ""},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("sheet.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
        }
    )
    assert response.status_code in [400, 422]

def test_similarity_rejects_invalid_pdf_magic_bytes(test_client):
    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Photosynthesis is the process..."},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("sheet.pdf", b"INVALID_BINARY_NOT_PDF", "application/pdf"),
        }
    )
    assert response.status_code == 422
    assert "not a valid PDF document" in response.json()["detail"]

def test_similarity_happy_path(test_client, monkeypatch):
    """Happy path with mocked PDF rasterization, OCR, and ML similarity."""
    dummy_page = Image.new("RGB", (400, 400), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Machine learning enables computers to learn from data."},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("student1.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "0" in data
    text_sim, diagram_sim, text, enc_diagram, enc_crop = data["0"]
    assert isinstance(text_sim, float)
    assert isinstance(diagram_sim, float)
    assert isinstance(text, str)

def test_concurrent_requests_isolated(test_client, monkeypatch):
    """Verifies that simultaneous requests with different inputs do not cross-contaminate (BUG-01)."""
    dummy_page = Image.new("RGB", (400, 400), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    def make_request(req_id: int):
        return test_client.post(
            "/similarity",
            data={"answer_key_text": f"Key text for request {req_id}"},
            files={
                "answer_key_diagram": (f"key_{req_id}.png", _create_png_bytes(), "image/png"),
                "answer_sheets": (f"sheet_{req_id}.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(make_request, 1)
        f2 = executor.submit(make_request, 2)
        r1 = f1.result()
        r2 = f2.result()

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert "0" in r1.json()
    assert "0" in r2.json()

def test_similarity_image_streaming_endpoints(test_client, monkeypatch, tmp_path):
    """PERF-08: Default mode returns lightweight image URLs and streams image bytes."""
    dummy_page = Image.new("RGB", (300, 300), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    d_path = str(tmp_path / "diagram.png")
    t_path = str(tmp_path / "text.png")
    Image.new("RGB", (50, 50), color="blue").save(d_path)
    Image.new("RGB", (50, 50), color="black").save(t_path)

    monkeypatch.setattr(
        "server.segment_lines_and_find_diagrams",
        lambda *args, **kwargs: {
            "segmented_folder": str(tmp_path),
            "text_crop_path": t_path,
            "diagram_path": d_path,
        }
    )
    monkeypatch.setattr("server.ocr_from_image", lambda *args, **kwargs: "Mocked OCR text")

    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Sample text"},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("student.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "0" in data
    text_sim, diagram_sim, text, diagram_url, text_crop_url = data["0"]
    assert diagram_url.startswith("/api/sessions/")
    assert text_crop_url.startswith("/api/sessions/")

    # Fetch streamed diagram image
    img_resp = test_client.get(diagram_url)
    assert img_resp.status_code == 200
    assert "image/" in img_resp.headers["content-type"]

    # Fetch streamed text crop image
    crop_resp = test_client.get(text_crop_url)
    assert crop_resp.status_code == 200
    assert "image/" in crop_resp.headers["content-type"]

def test_similarity_embed_base64_mode(test_client, monkeypatch, tmp_path):
    """PERF-08: Optional embed_base64=True returns inline base64 string for legacy callers."""
    dummy_page = Image.new("RGB", (300, 300), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    d_path = str(tmp_path / "diagram.png")
    t_path = str(tmp_path / "text.png")
    Image.new("RGB", (50, 50), color="blue").save(d_path)
    Image.new("RGB", (50, 50), color="black").save(t_path)

    monkeypatch.setattr(
        "server.segment_lines_and_find_diagrams",
        lambda *args, **kwargs: {
            "segmented_folder": str(tmp_path),
            "text_crop_path": t_path,
            "diagram_path": d_path,
        }
    )

    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Sample text", "embed_base64": "true"},
        files={
            "answer_key_diagram": ("key.png", _create_png_bytes(), "image/png"),
            "answer_sheets": ("student.pdf", _create_minimal_pdf_bytes(), "application/pdf"),
        }
    )
    assert response.status_code == 200
    data = response.json()
    _, _, _, diagram_ref, text_ref = data["0"]
    assert not diagram_ref.startswith("/api/sessions/")
    assert len(diagram_ref) > 0

def test_session_image_security_validation(test_client):
    """PERF-08: Invalid session IDs and image types are rejected."""
    # Invalid session ID format (path traversal attempt)
    resp = test_client.get("/api/sessions/../etc/passwd/sheets/0/diagram")
    assert resp.status_code in (400, 404)

    # Invalid image type
    resp2 = test_client.get(f"/api/sessions/{'a'*32}/sheets/0/unknown_type")
    assert resp2.status_code == 400

    # Non-existent session
    resp3 = test_client.get(f"/api/sessions/{'b'*32}/sheets/0/diagram")
    assert resp3.status_code == 404
