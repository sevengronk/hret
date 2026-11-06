"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hret.data.dataset import load_dataset  # noqa: E402
from hret.data.synthetic import generate_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    generate_synthetic(
        out,
        seed=7,
        num_users=40,
        num_businesses=15,
        num_services=12,
        num_regions=3,
        min_interactions=5,
        max_interactions=9,
    )
    return out


@pytest.fixture(scope="session")
def dataset(synthetic_dir: Path):
    return load_dataset(synthetic_dir)
