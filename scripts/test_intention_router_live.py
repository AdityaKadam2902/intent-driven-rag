"""
Run this manually, from the project root, with a real GEMINI_API_KEY:

    python scripts/test_intention_router_live.py

Makes REAL calls to Gemini to classify a set of test messages covering
all six intentions, through the ACTUAL compiled LangGraph graph - not
just the classification function in isolation.

What to check: does each message land in the intention category you'd
expect? The follow_up case specifically needs conversation_history to
be provided - check that "what about for interns" is correctly
classified as follow_up (not informational) when given the prior
turn's context.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.builder import build_graph

TEST_CASES = [
    ("hi", None, "chitchat"),
    ("thanks!", None, "chitchat"),
    ("what can you help with", None, "meta"),
    ("how confident are you in your answers", None, "meta"),
    ("what is the WFH policy", None, "informational"),
    ("who approves WFH requests", None, "informational"),
    (
        "what about for interns",
        [{"role": "assistant", "content": "The WFH policy requires Team Lead and Vikas Sir approval."}],
        "follow_up",
    ),
    ("apply leave for me", None, "actionable"),
    ("submit this WFH request on my behalf", None, "actionable"),
    ("I want to report harassment from my manager", None, "escalation"),
]

if __name__ == "__main__":
    graph = build_graph()
    correct = 0

    for message, history, expected in TEST_CASES:
        state = {"user_message": message}
        if history:
            state["conversation_history"] = history

        result = graph.invoke(state)
        actual = result["intention"]
        match = "OK" if actual == expected else "MISMATCH"
        if actual == expected:
            correct += 1

        print(f"[{match}] \"{message}\" -> expected={expected}, got={actual}")

    print(f"\n{correct}/{len(TEST_CASES)} classified as expected")
