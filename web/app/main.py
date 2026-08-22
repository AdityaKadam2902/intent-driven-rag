from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from web.app.auth.routes import router as auth_router
from web.app.chat.routes import router as chat_router
from web.app.config import get_settings
from web.app.dashboard.routes import router as dashboard_router
from web.app.dependencies import RedirectToLogin
from web.app.documents.routes import router as documents_router

settings = get_settings()

app = FastAPI(title="Meldep AgentPod")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie=settings.SESSION_COOKIE_NAME,
    max_age=settings.SESSION_MAX_AGE_SECONDS,
)

app.mount("/static", StaticFiles(directory="web/app/static"), name="static")


@app.exception_handler(RedirectToLogin)
def redirect_to_login(request: Request, exc: RedirectToLogin):
    return RedirectResponse(url="/login", status_code=303)


app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(documents_router)
