import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_USERNAME", "testuser")
os.environ.setdefault("APP_PASSWORD", "testpass")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-prod")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@localhost/meldep_sop_dev")

from web.app.config import get_settings
from web.app.main import app

get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def logged_in_client():
    c = TestClient(app)
    c.post("/login", data={"username": "testuser", "password": "testpass"})
    return c
