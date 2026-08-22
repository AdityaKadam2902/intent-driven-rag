from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from web.app.auth.session import is_authenticated, log_in, log_out
from web.app.config import Settings, get_settings

router = APIRouter()
templates = Jinja2Templates(directory="web/app/templates")


@router.get("/login")
def login_form(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"error": None}
    )


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
):
    if username == settings.APP_USERNAME and password == settings.APP_PASSWORD:
        log_in(request, user_id=username)
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": "Invalid username or password."},
        status_code=401,
    )


@router.post("/logout")
def logout(request: Request):
    log_out(request)
    return RedirectResponse(url="/login", status_code=303)
