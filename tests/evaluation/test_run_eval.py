import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.run_eval import check_case


def test_must_contain_all_passes_when_present():
    case = {"must_contain_all": ["Team Lead", "Vikas Sir"]}
    result = {"answer": "Approved by the Team Lead and Vikas Sir."}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_must_contain_all_fails_when_missing():
    case = {"must_contain_all": ["Team Lead", "Vikas Sir"]}
    result = {"answer": "Approved by the Team Lead."}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_must_not_contain_catches_softened_and():
    """The exact real bug this project cares most about: 'and' silently
    becoming 'or'."""
    case = {"must_not_contain": ["either", "or Vikas"]}
    result = {"answer": "Approved by either the Team Lead or Vikas Sir."}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_expected_intention_matches():
    case = {"expected_intention": "chitchat"}
    result = {"intention": "chitchat"}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_expected_intention_mismatch_fails():
    case = {"expected_intention": "chitchat"}
    result = {"intention": "informational"}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_confidence_threshold_passes_when_higher():
    case = {"expected_min_confidence": "Medium"}
    result = {"confidence": "High"}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_confidence_threshold_fails_when_lower():
    case = {"expected_min_confidence": "High"}
    result = {"confidence": "Low"}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_no_answer_case_passes_on_decline_language():
    case = {"expect_low_confidence_or_decline": True}
    result = {"answer": "I don't have information on this.", "confidence": "Medium"}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_no_answer_case_passes_on_low_confidence_alone():
    case = {"expect_low_confidence_or_decline": True}
    result = {"answer": "Here's a confident-sounding but wrong answer.", "confidence": "Low"}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_no_answer_case_fails_if_confident_and_no_decline_language():
    """The real failure mode this check exists to catch: a confident,
    fabricated-sounding answer to a question the SOPs don't cover."""
    case = {"expect_low_confidence_or_decline": True}
    result = {"answer": "The maternity leave policy grants 6 months paid leave.", "confidence": "High"}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_must_not_contain_source_passes_when_empty():
    case = {"must_not_contain_source": True}
    result = {"sources": []}
    passed, reasons = check_case(case, result)
    assert passed is True


def test_must_not_contain_source_fails_when_present():
    case = {"must_not_contain_source": True}
    result = {"sources": [{"document_title": "X"}]}
    passed, reasons = check_case(case, result)
    assert passed is False


def test_reasons_are_populated_for_every_check():
    """Failures must be diagnosable from the reasons list alone,
    without re-running anything."""
    case = {"expected_intention": "chitchat", "must_contain_all": ["hi"]}
    result = {"intention": "informational", "answer": "something else"}
    passed, reasons = check_case(case, result)
    assert len(reasons) == 2
    assert any("intention" in r for r in reasons)
    assert any("must contain" in r for r in reasons)


# ---------------------------------------------------------------------
# v2 additions: new checks, sequences, dataset validation, run() wiring
# ---------------------------------------------------------------------
from unittest.mock import patch

from evaluation import run_eval
from evaluation.run_eval import load_cases, normalize_steps, validate_cases


def test_must_contain_any_passes_when_one_alternative_present():
    case = {"must_contain_any": ["not allowed", "cannot"]}
    assert check_case(case, {"answer": "Interns cannot work from home."})[0] is True


def test_must_contain_any_fails_when_none_present():
    case = {"must_contain_any": ["not allowed", "cannot"]}
    assert check_case(case, {"answer": "Interns may work from home."})[0] is False


def test_min_distinct_source_documents():
    sources = [{"document_title": "SDLC Module"}, {"document_title": "Testing Process"},
               {"document_title": "Testing Process"}]
    assert check_case({"min_distinct_source_documents": 2}, {"sources": sources})[0] is True
    assert check_case({"min_distinct_source_documents": 3}, {"sources": sources})[0] is False


def test_expect_cache_hit_true_and_false():
    assert check_case({"expect_cache_hit": True}, {"cache_hit": True})[0] is True
    assert check_case({"expect_cache_hit": True}, {})[0] is False
    assert check_case({"expect_cache_hit": False}, {})[0] is True
    assert check_case({"expect_cache_hit": False}, {"cache_hit": True})[0] is False


def test_must_equal_previous():
    prev = {"answer": "a", "confidence": "High", "sources": [{"x": 1}]}
    same = {"answer": "a", "confidence": "High", "sources": [{"x": 1}]}
    assert check_case({"must_equal_previous": True}, same, prev)[0] is True
    assert check_case({"must_equal_previous": True}, {**same, "answer": "b"}, prev)[0] is False
    assert check_case({"must_equal_previous": True}, {**same, "confidence": "Low"}, prev)[0] is False


def test_must_differ_from_previous_catches_a_wrongly_served_cached_answer():
    """The entity-guard failure mode: step 2 returned step 1's answer."""
    prev = {"answer": "Vikas Sir and the Team Lead approve."}
    assert check_case({"must_differ_from_previous": True}, {"answer": "Management approval."}, prev)[0] is True
    assert check_case({"must_differ_from_previous": True}, dict(prev), prev)[0] is False


def test_previous_step_comparisons_fail_without_a_previous_step():
    assert check_case({"must_equal_previous": True}, {"answer": "a"})[0] is False
    assert check_case({"must_differ_from_previous": True}, {"answer": "a"})[0] is False


def test_normalize_steps_wraps_single_question_cases():
    single = {"id": "x", "question": "q"}
    seq = {"id": "y", "sequence": [{"question": "a"}, {"question": "b"}]}
    assert normalize_steps(single) == [single]
    assert normalize_steps(seq) == seq["sequence"]


def test_shipped_golden_dataset_is_valid():
    """Guards the real qa_pairs.json: no typos in check names, no
    duplicate ids, every case documents its target."""
    cases = load_cases()
    assert validate_cases(cases) == []
    assert len(cases) >= 17


def test_validate_catches_misspelled_check_key():
    bad = [{"id": "x", "targets": "t", "question": "q", "must_contain_al": ["a"]}]
    problems = validate_cases(bad)
    assert any("must_contain_al" in p for p in problems)


def test_validate_catches_duplicates_missing_targets_and_bad_sequences():
    cases = [
        {"id": "dup", "targets": "t", "question": "q"},
        {"id": "dup", "question": "q"},
        {"id": "seq", "targets": "t", "sequence": [{"question": "a", "must_equal_previous": True}]},
        {"id": "seq2", "targets": "t", "question": "stray", "sequence": [{"question": "a"}]},
    ]
    text = " | ".join(validate_cases(cases))
    assert "duplicate id: dup" in text
    assert "missing 'targets'" in text
    assert "first step" in text
    assert "unknown top-level key 'question'" in text


class _FakeGraph:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def invoke(self, state):
        self.asked.append(state["user_message"])
        return dict(self.answers[len(self.asked) - 1])


def _run_with(cases, answers, keep_cache=False):
    graph = _FakeGraph(answers)
    with patch.object(run_eval, "load_cases", return_value=cases), \
         patch.object(run_eval, "build_graph", return_value=graph), \
         patch.object(run_eval, "_flush_cache", return_value=3) as flush:
        counts = run_eval.run(keep_cache=keep_cache)
    return counts, graph, flush


def test_run_flushes_cache_by_default_and_not_with_keep_cache():
    case = {"id": "a", "targets": "t", "question": "q", "must_contain_all": ["x"]}
    _, _, flush = _run_with([case], [{"answer": "x"}])
    flush.assert_called_once()
    _, _, flush = _run_with([case], [{"answer": "x"}], keep_cache=True)
    flush.assert_not_called()


def test_run_executes_sequence_steps_in_order_and_chains_previous():
    case = {"id": "s", "targets": "t", "sequence": [
        {"question": "one", "must_contain_all": ["A"]},
        {"question": "two", "must_differ_from_previous": True},
    ]}
    counts, graph, _ = _run_with([case], [{"answer": "A"}, {"answer": "B"}])
    assert graph.asked == ["one", "two"]
    assert counts == (1, 1)


def test_run_fails_a_sequence_when_a_later_step_fails():
    case = {"id": "s", "targets": "t", "sequence": [
        {"question": "one"},
        {"question": "two", "must_differ_from_previous": True},
    ]}
    counts, _, _ = _run_with([case], [{"answer": "same"}, {"answer": "same"}])
    assert counts == (0, 1)


def test_run_refuses_to_start_on_an_invalid_dataset_before_any_graph_call():
    bad = {"id": "x", "targets": "t", "question": "q", "typo_key": 1}
    graph = _FakeGraph([])
    with patch.object(run_eval, "load_cases", return_value=[bad]), \
         patch.object(run_eval, "build_graph", return_value=graph):
        try:
            run_eval.run()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "typo_key" in str(e)
    assert graph.asked == []


def test_is_degraded_detects_both_breaker_fallbacks_and_not_normal_answers():
    assert run_eval.is_degraded({"answer": "I'm having trouble generating a written answer right now, but..."})
    assert run_eval.is_degraded({"answer": "The system is temporarily unable to search the SOPs right now"})
    assert not run_eval.is_degraded({"answer": "Team Lead and Vikas Sir approve."})
    assert not run_eval.is_degraded({})


def test_run_pauses_between_steps_but_not_before_the_first():
    cases = [
        {"id": "a", "targets": "t", "question": "q1"},
        {"id": "b", "targets": "t", "sequence": [{"question": "q2"}, {"question": "q3"}]},
    ]
    graph = _FakeGraph([{"answer": "x"}] * 3)
    with patch.object(run_eval, "load_cases", return_value=cases), \
         patch.object(run_eval, "build_graph", return_value=graph), \
         patch.object(run_eval, "_flush_cache", return_value=0), \
         patch.object(run_eval.time, "sleep") as sleep:
        run_eval.run(pause_seconds=7)
    assert sleep.call_count == 2                       # 3 questions -> 2 gaps
    assert all(c.args == (7,) for c in sleep.call_args_list)


def test_run_does_not_sleep_by_default():
    graph = _FakeGraph([{"answer": "x"}] * 2)
    cases = [{"id": "a", "targets": "t", "question": "q1"}, {"id": "b", "targets": "t", "question": "q2"}]
    with patch.object(run_eval, "load_cases", return_value=cases), \
         patch.object(run_eval, "build_graph", return_value=graph), \
         patch.object(run_eval, "_flush_cache", return_value=0), \
         patch.object(run_eval.time, "sleep") as sleep:
        run_eval.run()
    sleep.assert_not_called()


def test_run_warns_when_answers_were_degraded(capsys):
    case = {"id": "a", "targets": "t", "question": "q", "must_contain_all": ["Vikas"]}
    degraded = {"answer": "I'm having trouble generating a written answer right now"}
    _run_with([case], [degraded])
    out = capsys.readouterr().out
    assert "degraded fallback answer" in out
    assert "NOT a reliable quality measurement" in out
