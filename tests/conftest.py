import shutil
from pathlib import Path

import pytest


@pytest.fixture
def test_data_tmp_dir() -> Path:
    test_data_tmp_dir = Path(__file__).parent / "data_tmp"
    test_data_tmp_dir.mkdir(exist_ok=True)
    yield test_data_tmp_dir
    shutil.rmtree(test_data_tmp_dir)
