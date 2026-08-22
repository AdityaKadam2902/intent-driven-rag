from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
_ALLOWED_TAGS = ["p", "strong", "em", "ul", "ol", "li", "br", "code", "pre", "a"]
_ALLOWED_ATTRS = {"a": ["href", "title", "rel"]}

from web.app.assistants.registry import get_assistant
from web.app.auth.session import chat_session_id
from web.app.chat.history import (
    append_assistant_turn,
    append_user_turn,
    clear_history,
    get_conversation_history,
    get_display_turns,
)
from web.app.dependencies import require_login
import bleach
import markdown as md
from markupsafe import Markup

router = APIRouter(prefix="/chat")
templates = Jinja2Templates(directory="web/app/templates")

def render_markdown(text: str) -> Markup:
    if not text:
        return Markup("")
    html = md.markdown(text, extensions=["extra", "nl2br"])
    clean = bleach.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS)
    return Markup(clean)


templates.env.filters["render_markdown"] = render_markdown

@router.get("/{assistant_id}")
def chat_page(
    request: Request,
    assistant_id: str,
    _user_id: str = Depends(require_login),
):
    try:
        assistant = get_assistant(assistant_id)
    except ValueError:
        return RedirectResponse(url="/", status_code=303)

    session_id = chat_session_id(request)
    turns = get_display_turns(session_id, assistant_id)

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "assistant": assistant,
            "turns": turns,
            "error": None,
        },
    )


@router.post("/{assistant_id}/clear")
def chat_clear(
    request: Request,
    assistant_id: str,
    _user_id: str = Depends(require_login),
):
    session_id = chat_session_id(request)
    clear_history(session_id, assistant_id)
    return RedirectResponse(url=f"/chat/{assistant_id}", status_code=303)


@router.post("/{assistant_id}")
def chat_ask(
    request: Request,
    assistant_id: str,
    message: str = Form(...),
    _user_id: str = Depends(require_login),
):
    try:
        assistant = get_assistant(assistant_id)
    except ValueError:
        return RedirectResponse(url="/", status_code=303)

    session_id = chat_session_id(request)
    message = message.strip()

    error = None
    if message:
        history = get_conversation_history(session_id, assistant_id)
        append_user_turn(session_id, assistant_id, message)

        try:
            graph = assistant.build_graph()
            result = graph.invoke(
                {"user_message": message, "conversation_history": history}
            )
            append_assistant_turn(
                session_id,
                assistant_id,
                content=result.get("answer", ""),
                confidence=result.get("confidence"),
                sources=result.get("sources"),
                intention=result.get("intention"),
            )
        except Exception as exc:  # RAG call failed (Gemini/Postgres/etc.)
            error = f"Something went wrong answering that question: {exc}"

    turns = get_display_turns(session_id, assistant_id)

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "assistant": assistant,
            "turns": turns,
            "error": error,
        },
    )