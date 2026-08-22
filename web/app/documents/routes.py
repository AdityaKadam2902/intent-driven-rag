import os
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from web.app.assistants.registry import get_assistant
from web.app.config import get_settings
from web.app.dependencies import require_login
from web.app.documents.ingestion_status import (
    get_latest_run,
    mark_failed,
    mark_success,
    start_run,
)

router = APIRouter(prefix="/documents")
templates = Jinja2Templates(directory="web/app/templates")


def _staging_dir(assistant_id: str) -> str:
    settings = get_settings()
    path = os.path.join(settings.UPLOAD_STAGING_DIR, assistant_id)
    os.makedirs(path, exist_ok=True)
    return path


def _run_ingestion(assistant_id: str, run_id: int) -> None:
    assistant = get_assistant(assistant_id)
    try:
        summary = assistant.ingest_directory(_staging_dir(assistant_id))
        processed = summary.get("inserted", 0) + summary.get("updated", 0)
        errors = summary.get("errors") or []
        if errors:
            mark_failed(run_id, "; ".join(errors))
        else:
            mark_success(run_id, processed)
    except Exception as exc:
        mark_failed(run_id, str(exc))


@router.get("/{assistant_id}")
def documents_page(
    request: Request,
    assistant_id: str,
    user_id: str = Depends(require_login),
):
    try:
        assistant = get_assistant(assistant_id)
    except ValueError:
        return RedirectResponse(url="/", status_code=303)

    latest_run = get_latest_run(assistant_id)
    documents = assistant.list_documents()
    staged_files = sorted(os.listdir(_staging_dir(assistant_id)))

    return templates.TemplateResponse(
        request,
        "documents.html",
        {
            "assistant": assistant,
            "documents": documents,
            "staged_files": staged_files,
            "latest_run": latest_run,
            "error": None,
        },
    )


@router.post("/{assistant_id}/upload")
def upload_documents(
    request: Request,
    assistant_id: str,
    files: list[UploadFile],
    user_id: str = Depends(require_login),
):
    try:
        get_assistant(assistant_id)
    except ValueError:
        return RedirectResponse(url="/", status_code=303)

    staging = _staging_dir(assistant_id)
    for f in files:
        if not f.filename.endswith(".txt"):
            continue
        dest = os.path.join(staging, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    return RedirectResponse(url=f"/documents/{assistant_id}", status_code=303)


@router.post("/{assistant_id}/ingest")
def trigger_ingestion(
    request: Request,
    assistant_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_login),
):
    try:
        get_assistant(assistant_id)
    except ValueError:
        return RedirectResponse(url="/", status_code=303)

    run_id = start_run(assistant_id)
    background_tasks.add_task(_run_ingestion, assistant_id, run_id)

    return RedirectResponse(url=f"/documents/{assistant_id}", status_code=303)
