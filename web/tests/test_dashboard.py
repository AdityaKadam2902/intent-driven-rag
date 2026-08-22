def test_dashboard_renders_sop_assistant_tile(logged_in_client):
    resp = logged_in_client.get("/")
    assert resp.status_code == 200
    assert "SOP Assistant" in resp.text
    assert "/chat/sop_assistant" in resp.text
    assert "/documents/sop_assistant" in resp.text


def test_dashboard_shows_username_in_nav(logged_in_client):
    resp = logged_in_client.get("/")
    assert "testuser" in resp.text
