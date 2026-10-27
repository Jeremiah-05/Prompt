"""
Pytest fixtures for isolating test execution from the live database.
Every test runs against a fresh temporary copy of data/shopease.db.
"""

import shutil
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
LIVE_DB_PATH = ROOT_DIR / "data" / "shopease.db"


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path, monkeypatch):
    """Ensure every test runs against an isolated copy of data/shopease.db.
    
    Prevents any test from ever modifying or mutating the live database.
    """
    import tools
    import data.seed_db
    import eval.test_cases

    temp_db_path = tmp_path / "shopease_isolated.db"

    if LIVE_DB_PATH.exists():
        shutil.copy2(LIVE_DB_PATH, temp_db_path)
    else:
        data.seed_db.init_db(temp_db_path, force=True)

    monkeypatch.setattr(tools, "DB_PATH", temp_db_path)
    monkeypatch.setattr(data.seed_db, "DB_PATH", temp_db_path)
    monkeypatch.setattr(eval.test_cases, "DB_PATH", temp_db_path)

    yield temp_db_path
