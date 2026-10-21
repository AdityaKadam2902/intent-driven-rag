# Change Understanding Docs

Every significant change to this project gets a numbered document in this folder,
written so that both a technical reader and a non-technical reader can follow it
end to end: **what changed, why, how it works, what it affects, what it costs, and
what could go wrong.**

| # | Date | Title | Status | Files affected (summary) |
|---|---|---|---|---|
| 001 | 2026-10-21 | [Two-Layer Semantic Cache (Redis) for the Query Pipeline](./001-caching-layer.md) | Written, **not yet committed** | `app/cache/*` (new), `app/graph/nodes/cache/*` (new), `app/graph/builder.py`, `app/graph/state.py`, `app/config/engine_config.py`, `app/repositories/document_versions.py` (new), `tests/cache/*` (new), `tests/graph/conftest.py` (new), `requirements.txt` |

## Naming convention

```
docs/changes/NNN-short-kebab-title.md
```

`NNN` is a zero-padded sequence number, so the files sort correctly and the order of
changes is always obvious.

## Template for a new change doc

Each document should contain, at minimum:

1. **TL;DR** — what changed, in one paragraph.
2. **Why** — the problem being solved (business + technical).
3. **Concepts explained** — plain English first, then the technical detail.
4. **File-by-file map** — new files vs modified files, with the change in each.
5. **Step-by-step flow** — before/after diagrams and the exact execution order.
6. **Non-technical summary** — for teammates who do not read the code.
7. **Failure matrix** — "what happens if X breaks?".
8. **Configuration** — every new setting, default and env var.
9. **Tests** — what is covered and what it proves.
10. **Known gaps and risks** — including deliberate simplifications.
11. **Cheat sheet** — the mental model and where to look in the code.

---

## Index of concepts explained so far

- **Cache** — a scratch pad for expensive work; always an optimisation, never a dependency.
- **Two-layer cache** — Layer 0 = exact match (cheap, zero risk); Layer 1 = semantic match (smart, guarded).
- **Embedding / cosine similarity** — how "similar meaning" is measured numerically.
- **The entity guard** — the rule that stops a semantically-similar question from
  receiving a confidently wrong answer.
- **Version-tied invalidation** — how the cache stays fresh without any manual purging.
- **Circuit breaker** — the 3-state guard (`CLOSED` / `OPEN` / `HALF_OPEN`) that keeps a
  flaky dependency from taking the system down with it.
- **Fail-open vs fail-closed** — the rule that caching failures degrade to "no cache",
  never to "no answer".
- **Write-through** — writing the answer into the cache immediately after producing it.
