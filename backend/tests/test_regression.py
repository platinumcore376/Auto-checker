import io
import pytest
from PIL import Image
from Utils.image_similarity import image_similarity
from Utils.ocr import _extract_sort_index

def _create_minimal_pdf():
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] >> endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
        b"trailer << /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )

def _create_png():
    buf = io.BytesIO()
    img = Image.new("RGB", (50, 50), color="white")
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_regression_bug_08_empty_diagram_list_no_crash():
    """BUG-08: torch.stack([]) crashed on empty list; now returns []."""
    res = image_similarity(object(), [])
    assert res == []

def test_regression_bug_07_ocr_filename_sorting():
    """BUG-07: Filenames like line_10.png vs line_2.png or unexpected names crashed split('_')[1]."""
    files = ["line_10.png", "line_1.png", "line_2.png", "other_file.png"]
    sorted_files = sorted(files, key=_extract_sort_index)
    assert sorted_files[0] == "line_1.png"
    assert sorted_files[1] == "line_2.png"
    assert sorted_files[2] == "line_10.png"
    assert sorted_files[3] == "other_file.png"

def test_regression_bug_03_no_diagram_sheet_in_api(test_client, monkeypatch):
    """BUG-03: Sheets without diagrams crashed with IndexError; now handled with diagram_score=0.0."""
    dummy_page = Image.new("RGB", (300, 300), color="white")
    monkeypatch.setattr("server.convert_from_bytes", lambda *args, **kwargs: [dummy_page])

    # Mock segmentation to return no diagram
    monkeypatch.setattr(
        "server.segment_lines_and_find_diagrams",
        lambda *args, **kwargs: {
            "segmented_folder": None,
            "text_crop_path": None,
            "diagram_path": None,
        }
    )

    response = test_client.post(
        "/similarity",
        data={"answer_key_text": "Sample answer text."},
        files={
            "answer_key_diagram": ("key.png", _create_png(), "image/png"),
            "answer_sheets": ("student_no_diagram.pdf", _create_minimal_pdf(), "application/pdf"),
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "0" in data
    # Diagram score must be 0.0, encoded diagram empty string, no crash
    text_sim, diagram_sim, text, enc_diagram, enc_crop = data["0"]
    assert diagram_sim == 0.0
    assert enc_diagram == ""
