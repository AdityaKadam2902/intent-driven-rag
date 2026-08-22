from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from web.app.assistants.registry import ASSISTANT_REGISTRY
from web.app.dependencies import require_login

router = APIRouter()
templates = Jinja2Templates(directory="web/app/templates")


@router.get("/")
def dashboard(request: Request, user_id: str = Depends(require_login)):
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user_id": user_id,
            "assistants": list(ASSISTANT_REGISTRY.values()),
        },
    )
