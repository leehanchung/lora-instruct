import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_cwd(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
