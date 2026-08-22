"""
Run this manually, from the project root, AFTER running
scripts/run_ingestion.py at least once. Requires your real
GEMINI_API_KEY and Postgres connection in .env.

    python scripts/test_full_pipeline_live.py

This is the real, complete end-to-end proof: a question goes in, and
comes out the other side as an actual synthesized answer with
citations and a confidence level - through the full compiled graph
(classify -> retrieve -> rerank -> synthesize), using your real
ingested SOP data and real Gemini calls at every step.

What to check: for "who approves WFH requests", does the answer
correctly say BOTH the Team Lead and Vikas Sir are needed (not "or"),
and is the confidence High? For the follow-up example, does it
correctly resolve "what about for interns" using the prior turn?
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.builder import build_graph


def ask(question, history=None):
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}")

    state = {"user_message": question}
    if history:
        state["conversation_history"] = history

    result = build_graph().invoke(state)

    print(f"Intention: {result['intention']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['document_title']} v{s['document_version']} ({s['heading_path']})")

    return result


if __name__ == "__main__":
    ask("hi")
    ask("who approves WFH requests")
    result = ask("what is the WFH policy")
    ask(
        "what about for interns",
        history=[{"role": "assistant", "content": result.get("answer", "")}],
    )
    ask("apply leave for me")
