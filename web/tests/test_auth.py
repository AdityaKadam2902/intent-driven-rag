def test_login_page_renders(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "Log in" in resp.text or "Username" in resp.text


def test_login_success_redirects_to_dashboard(client):
    resp = client.post(
        "/login",
        data={"username": "testuser", "password": "testpass"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


def test_login_failure_shows_error(client):
    resp = client.post(
        "/login",
        data={"username": "testuser", "password": "wrongpass"},
    )
    assert resp.status_code == 401
    assert "Invalid username or password" in resp.text


def test_dashboard_requires_login(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_logout_clears_session(logged_in_client):
    resp = logged_in_client.get("/")
    assert resp.status_code == 200

    logged_in_client.post("/logout")
    resp = logged_in_client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"
