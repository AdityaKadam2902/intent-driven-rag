import io
import os
import shutil

import pytest

import web.app.documents.routes as documents_routes
from web.app.config import get_settings


class _FakeRunStore:
    def __init__(self):
        self.runs = {}  # assistant_id -> dict
        self._next_id = 1

    def start_run(self, assistant_id):
        run_id = self._next_id
        self._next_id += 1
        self.runs[assistant_id] = {
            "id": run_id,
            "status": "running",
            "doc_count": None,
            "error_message": None,
        }
        return run_id

    def mark_success(self, run_id, doc_count):
        for run in self.runs.values():
            if run["id"] == run_id:
                run["status"] = "success"
                run["doc_count"] = doc_count

    def mark_failed(self, run_id, error_message):
        for run in self.runs.values():
            if run["id"] == run_id:
                run["status"] = "failed"
                run["error_message"] = error_message

    def get_latest_run(self, assistant_id):
        return self.runs.get(assistant_id)


@pytest.fixture
def fake_run_store(monkeypatch):
    store = _FakeRunStore()
    monkeypatch.setattr(documents_routes, "start_run", store.start_run)
    monkeypatch.setattr(documents_routes, "mark_success", store.mark_success)
    monkeypatch.setattr(documents_routes, "mark_failed", store.mark_failed)
    monkeypatch.setattr(documents_routes, "get_latest_run", store.get_latest_run)
    return store


@pytest.fixture(autouse=True)
def clean_staging_dir():
    settings = get_settings()
    yield
    if os.path.isdir(settings.UPLOAD_STAGING_DIR):
        shutil.rmtree(settings.UPLOAD_STAGING_DIR)


def test_documents_page_requires_login(client, fake_run_store):
    resp = client.get("/documents/sop_assistant", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_documents_page_unknown_assistant_redirects_home(logged_in_client, fake_run_store):
    resp = logged_in_client.get("/documents/does_not_exist", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


def test_documents_page_renders_ingested_list(logged_in_client, fake_run_store):
    resp = logged_in_client.get("/documents/sop_assistant")
    assert resp.status_code == 200
    assert "Work From Home (WFH) Policy" in resp.text
    assert "No files staged yet" in resp.text


def test_upload_stages_txt_file(logged_in_client, fake_run_store):
    upload = {"files": ("policy.txt", io.BytesIO(b"some SOP content"), "text/plain")}
    resp = logged_in_client.post(
        "/documents/sop_assistant/upload", files=upload, follow_redirects=False
    )
    assert resp.status_code == 303

    resp = logged_in_client.get("/documents/sop_assistant")
    assert "policy.txt" in resp.text


def test_upload_ignores_non_txt_files(logged_in_client, fake_run_store):
    upload = {"files": ("notes.pdf", io.BytesIO(b"binary junk"), "application/pdf")}
    logged_in_client.post("/documents/sop_assistant/upload", files=upload)

    resp = logged_in_client.get("/documents/sop_assistant")
    assert "notes.pdf" not in resp.text
    assert "No files staged yet" in resp.text


def test_ingest_triggers_background_task_and_shows_running_status(
    logged_in_client, fake_run_store
):
    upload = {"files": ("policy.txt", io.BytesIO(b"content"), "text/plain")}
    logged_in_client.post("/documents/sop_assistant/upload", files=upload)

    resp = logged_in_client.post(
        "/documents/sop_assistant/ingest", follow_redirects=False
    )
    assert resp.status_code == 303

    # TestClient runs BackgroundTasks synchronously before the response
    # is returned, so by the time we check status it should already be
    # marked success (using the stub ingest_directory, which succeeds).
    resp = logged_in_client.get("/documents/sop_assistant")
    assert "succeeded" in resp.text or "in progress" in resp.text
