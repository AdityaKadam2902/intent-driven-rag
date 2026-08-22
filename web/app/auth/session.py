import uuid

from starlette.requests import Request

USER_KEY = "user_id"
CHAT_SESSION_KEY = "chat_session_id"


def log_in(request: Request, user_id: str) -> None:
    request.session[USER_KEY] = user_id
    # stable id used to group rows in chat_turns; independent of auth itself
    request.session.setdefault(CHAT_SESSION_KEY, str(uuid.uuid4()))


def log_out(request: Request) -> None:
    request.session.clear()


def current_user(request: Request) -> str | None:
    return request.session.get(USER_KEY)


def is_authenticated(request: Request) -> bool:
    return current_user(request) is not None


def chat_session_id(request: Request) -> str:
    # defensive fallback; should always be set by log_in
    if CHAT_SESSION_KEY not in request.session:
        request.session[CHAT_SESSION_KEY] = str(uuid.uuid4())
    return request.session[CHAT_SESSION_KEY]
