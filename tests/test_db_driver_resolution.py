import pytest
from sqlalchemy.engine import make_url

from smartkcet.db.session import _resolve_database_url


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:secret@db.example.test:5432/sample",
        "postgresql+psycopg2://user:secret@db.example.test:5432/sample",
    ],
)
def test_postgresql_urls_use_pg8000(database_url, monkeypatch):
    monkeypatch.delenv("USE_SQLITE", raising=False)
    monkeypatch.setenv("DATABASE_URL", database_url)

    resolved_url = make_url(_resolve_database_url())

    assert resolved_url.drivername == "postgresql+pg8000"
    assert resolved_url.username == "user"
    assert resolved_url.password == "secret"
    assert resolved_url.host == "db.example.test"
    assert resolved_url.database == "sample"


def test_explicit_other_database_url_is_preserved(monkeypatch):
    monkeypatch.delenv("USE_SQLITE", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    assert _resolve_database_url() == "sqlite:///:memory:"


def test_use_sqlite_overrides_database_url(monkeypatch):
    monkeypatch.setenv("USE_SQLITE", "1")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg2://user:secret@db.example.test/sample"
    )

    assert _resolve_database_url().startswith("sqlite:///")