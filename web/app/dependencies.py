from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from web.app.auth.session import is_authenticated


class RedirectToLogin(StarletteHTTPException):
    """Raised by require_login; handled by an exception handler in main.py
    so route functions can stay simple and just declare the dependency."""

    def __init__(self):
        super().__init__(status_code=303)


def require_login(request: Request) -> str:
    from web.app.auth.session import current_user

    if not is_authenticated(request):
        raise RedirectToLogin()
    return current_user(request)
