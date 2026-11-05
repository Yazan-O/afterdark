import pytest


@pytest.fixture(autouse=True)
def _temp_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("AFTERDARK_SPEND", str(tmp_path / "spend.csv"))
