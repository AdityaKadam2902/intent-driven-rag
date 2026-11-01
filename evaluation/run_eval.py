"""
Runs the golden dataset (evaluation/golden_dataset/qa_pairs.json) through
the REAL compiled graph - real Gemini calls, real Postgres retrieval,
real Redis cache. Needs scripts/run_ingestion.py to have been run, a
GEMINI_API_KEY in .env, and (for the cache cases) Redis running.

    python evaluation/run_eval.py              # clears cache first
    python evaluation/run_eval.py --keep-cache # leave existing cache entries alone
    python evaluation/run_eval.py --pause 12   # free tier: ~15 Gemini requests/minute

By default the run starts by deleting every `cache:*` key in Redis, so
the run measures the CURRENT code against COLD cache. Without this, a
stale cached answer from before a prompt/chunker change would be served
instantly and the evaluation would silently grade yesterday's pipeline.

A case is either a single question (checks live on the case itself) or
a `sequence` of steps run in order in the same process, so cache
behaviour across questions can be asserted.

Checks (all optional per case/step; every performed check is reported):
  expected_intention             router label must match
  must_contain_all               every phrase present (case-insensitive)
  must_contain_any               at least one phrase present - for answers
                                 whose wording legitimately varies
  must_not_contain               none present
  expected_min_confidence        High > Medium > Low
  expect_low_confidence_or_decline  honest "I don't know"
  must_not_contain_source        no SOP sources shown (escalation)
  min_distinct_source_documents  sources span at least N different documents
  expect_cache_hit               true/false - did the cache serve this step
  must_equal_previous            answer+confidence+sources identical to prior step
  must_differ_from_previous      answer differs from prior step
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.builder import build_graph

CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}
DECLINE_PHRASES = ["don't have", "not covered", "no information", "not sure", "check with"]

# Answers the pipeline produces when a circuit breaker is open. A check
# failing on one of these says nothing about answer quality - it means
# Gemini or Postgres was unavailable (typically free-tier rate limiting).
DEGRADED_MARKERS = ["trouble generating a written answer", "temporarily unable to search"]


def is_degraded(result):
    answer = (result.get("answer") or "").lower()
    return any(m in answer for m in DEGRADED_MARKERS)


def load_cases():
    path = os.path.join(os.path.dirname(__file__), "golden_dataset", "qa_pairs.json")
    with open(path) as f:
        data = json.load(f)
    return [c for c in data["cases"] if not c.get("skip")]


def normalize_steps(case):
    """A single-question case is just a one-step sequence whose checks
    sit on the case itself."""
    return case["sequence"] if "sequence" in case else [case]


CHECK_KEYS = {
    "expected_intention", "must_contain_all", "must_contain_any", "must_not_contain",
    "expected_min_confidence", "expect_low_confidence_or_decline", "must_not_contain_source",
    "min_distinct_source_documents", "expect_cache_hit", "must_equal_previous",
    "must_differ_from_previous",
}
STEP_STRUCTURAL_KEYS = {"id", "targets", "question", "requires_history", "skip"}
SEQUENCE_CASE_KEYS = {"id", "targets", "sequence", "skip"}


def validate_cases(cases):
    """Returns a list of problems. Runs before any API call is made: a
    misspelled check key (e.g. 'must_contain_al') would otherwise be
    silently ignored and the case would pass without checking anything."""
    problems, seen = [], set()
    for case in cases:
        cid = case.get("id", "<no id>")
        if cid in seen:
            problems.append(f"duplicate id: {cid}")
        seen.add(cid)
        if "targets" not in case:
            problems.append(f"{cid}: missing 'targets'")
        if "sequence" in case:
            for key in case:
                if key not in SEQUENCE_CASE_KEYS:
                    problems.append(f"{cid}: unknown top-level key '{key}' in a sequence case")
        for i, step in enumerate(normalize_steps(case), 1):
            where = f"{cid} step {i}" if "sequence" in case else cid
            if "question" not in step:
                problems.append(f"{where}: missing 'question'")
            for key in step:
                if key not in CHECK_KEYS | STEP_STRUCTURAL_KEYS:
                    problems.append(f"{where}: unknown key '{key}'")
            if i == 1 and (step.get("must_equal_previous") or step.get("must_differ_from_previous")):
                problems.append(f"{where}: compares to a previous step but is the first step")
    return problems


def _flush_cache():
    """Delete every cache:* key. Returns how many were removed, or None
    if Redis could not be reached (the run continues either way - the
    cache is an optimization, and cache-specific steps will simply
    report failures)."""
    try:
        from app.cache.redis_client import get_redis_client

        client = get_redis_client()
        keys = list(client.scan_iter("cache:*"))
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception:
        return None


def check_case(case, result, previous=None):
    """Returns (passed, reasons). `previous` is the prior step's result
    for sequence steps. Reasons describe every check performed, pass or
    fail, so a failure is diagnosable without re-running."""
    reasons = []
    passed = True
    answer = result.get("answer", "") or ""
    lowered = answer.lower()
    intention = result.get("intention")
    confidence = result.get("confidence")
    sources = result.get("sources", []) or []

    def record(ok, text):
        nonlocal passed
        passed = passed and ok
        reasons.append(f"{'OK' if ok else 'FAIL'} {text}")

    if "expected_intention" in case:
        record(intention == case["expected_intention"],
               f"intention: expected={case['expected_intention']} got={intention}")

    for phrase in case.get("must_contain_all", []):
        record(phrase.lower() in lowered, f'must contain: "{phrase}"')

    if "must_contain_any" in case:
        found = [p for p in case["must_contain_any"] if p.lower() in lowered]
        record(bool(found), f"must contain at least one of {case['must_contain_any']}")

    for phrase in case.get("must_not_contain", []):
        record(phrase.lower() not in lowered, f'must NOT contain: "{phrase}"')

    if "expected_min_confidence" in case:
        ok = CONFIDENCE_RANK.get(confidence, -1) >= CONFIDENCE_RANK.get(case["expected_min_confidence"], 0)
        record(ok, f"confidence >= {case['expected_min_confidence']}: got={confidence}")

    if case.get("expect_low_confidence_or_decline"):
        declined = any(p in lowered for p in DECLINE_PHRASES)
        record(declined or confidence == "Low",
               f"honest about not knowing: confidence={confidence}, declined_language={declined}")

    if case.get("must_not_contain_source"):
        record(len(sources) == 0, f"no SOP sources shown: sources={len(sources)}")

    if "min_distinct_source_documents" in case:
        docs = {s.get("document_title") for s in sources}
        record(len(docs) >= case["min_distinct_source_documents"],
               f"sources span >= {case['min_distinct_source_documents']} documents: got {sorted(d for d in docs if d)}")

    if "expect_cache_hit" in case:
        actual = bool(result.get("cache_hit"))
        record(actual == case["expect_cache_hit"],
               f"cache_hit: expected={case['expect_cache_hit']} got={actual}")

    if case.get("must_equal_previous"):
        if previous is None:
            record(False, "must_equal_previous needs a previous step")
        else:
            same = (answer == (previous.get("answer") or "")
                    and confidence == previous.get("confidence")
                    and sources == (previous.get("sources") or []))
            record(same, "answer, confidence and sources identical to previous step")

    if case.get("must_differ_from_previous"):
        if previous is None:
            record(False, "must_differ_from_previous needs a previous step")
        else:
            record(answer != (previous.get("answer") or ""), "answer differs from previous step")

    return passed, reasons


def run(keep_cache=False, pause_seconds=0.0):
    cases = load_cases()
    problems = validate_cases(cases)
    if problems:
        raise ValueError("Invalid golden dataset:\n  " + "\n  ".join(problems))
    graph = build_graph()

    flushed = None if keep_cache else _flush_cache()

    outcomes = []
    first_call = True
    for case in cases:
        previous = None
        steps_out = []
        case_passed = True
        for step in normalize_steps(case):
            if pause_seconds and not first_call:
                time.sleep(pause_seconds)   # stay under the Gemini free-tier rate limit
            first_call = False
            state = {"user_message": step["question"]}
            if "requires_history" in step:
                state["conversation_history"] = step["requires_history"]
            result = graph.invoke(state)
            ok, reasons = check_case(step, result, previous)
            case_passed = case_passed and ok
            steps_out.append((step, ok, reasons, result))
            previous = result
        outcomes.append((case, case_passed, steps_out))

    print(f"\n{'=' * 70}\nGOLDEN DATASET RESULTS\n{'=' * 70}")
    if keep_cache:
        print("Cache: kept as-is (--keep-cache) - results may reflect older pipeline versions")
    elif flushed is None:
        print("Cache: Redis unreachable, nothing flushed - cache steps will likely fail")
    else:
        print(f"Cache: cleared {flushed} cache:* key(s) before running (cold start)")

    passed_count = 0
    degraded = 0
    for case, case_passed, steps_out in outcomes:
        passed_count += case_passed
        print(f"\n[{'PASS' if case_passed else 'FAIL'}] {case['id']}")
        print(f"  Targets: {case['targets']}")
        for i, (step, ok, reasons, result) in enumerate(steps_out, 1):
            if len(steps_out) > 1:
                print(f"  step {i}: {step['question']!r}")
            else:
                print(f"  Q: {step['question']}")
            for r in reasons:
                print(f"    {r}")
            if is_degraded(result):
                degraded += 1
                print("    NOTE: degraded fallback answer (a circuit breaker was open) - "
                      "infrastructure, not answer quality")
            if not ok:
                print(f"    Full answer was: {(result.get('answer') or '')[:300]}")

    print(f"\n{'=' * 70}\n{passed_count}/{len(outcomes)} passed\n{'=' * 70}")
    if degraded:
        print(f"WARNING: {degraded} step(s) returned a degraded fallback answer, so this run is NOT "
              f"a reliable quality measurement.\nRe-run with --pause 12 (free-tier rate limit) once the "
              f"Gemini breaker has recovered (~30s).")
    return passed_count, len(outcomes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-cache", action="store_true",
                        help="do not clear cache:* keys before running")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to wait between questions; use 12 on the Gemini free tier")
    args = parser.parse_args()
    run(keep_cache=args.keep_cache, pause_seconds=args.pause)
