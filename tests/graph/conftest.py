"""
The graph now contains cache_lookup / cache_write nodes. Tests that
exercise routing and the retrieval path shouldn't depend on a live
Redis/Postgres/embedder, so by default both cache nodes are replaced
with pass-throughs (always a miss, never a write).

Tests that want the REAL cache nodes (with fakeredis) opt out with
@pytest.mark.real_cache - see test_builder_cache.py.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "real_cache: use the real cache nodes instead of pass-throughs")


@pytest.fixture(autouse=True)
def _stub_cache_nodes(request, monkeypatch):
    if request.node.get_closest_marker("real_cache"):
        return
    monkeypatch.setattr("app.graph.builder.cache_lookup", lambda state: {"cache_hit": False})
    monkeypatch.setattr("app.graph.builder.cache_write", lambda state: {})
