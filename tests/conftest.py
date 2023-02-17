import shutil
from pathlib import Path
from typing import Generator, TypeVar

import pytest

T = TypeVar("T")
YieldFixture = Generator[T, None, None]


@pytest.fixture(scope="module")
def test_data_tmp_dir() -> YieldFixture[Path]:
    test_data_tmp_dir = Path(__file__).parent / "data_tmp"
    test_data_tmp_dir.mkdir(exist_ok=True)
    yield test_data_tmp_dir
    shutil.rmtree(test_data_tmp_dir)


@pytest.fixture(scope="module")
def test_data_dir() -> Path:
    return Path(__file__).parent / "data"
