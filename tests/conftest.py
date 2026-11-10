"""Shared pytest fixtures."""
import os
import tempfile

os.environ.setdefault("ROUTELIX_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ROUTELIX_ADMIN_USER", "admin")
os.environ.setdefault("ROUTELIX_ADMIN_PASSWORD", "test-password-123")
os.environ.setdefault("ROUTELIX_COOKIE_SECURE", "0")
os.environ.setdefault(
    "ROUTELIX_USERS_FILE",
    os.path.join(tempfile.mkdtemp(prefix="routelix-test-users-"), "users.json"),
)

import pytest


@pytest.fixture(autouse=True)
def _reset_groq_pool_singleton():
    import src.groq_pool as gp

    gp._pool_singleton = None
    yield
    gp._pool_singleton = None
