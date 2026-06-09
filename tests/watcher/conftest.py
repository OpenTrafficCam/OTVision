import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _isolate_locks(tmp_path_factory, monkeypatch):
    """Keep per-camera lock files out of the repo during tests."""
    import track_continuous as tc

    monkeypatch.setattr(tc, "LOCK_DIR", tmp_path_factory.mktemp("locks"))
