from __future__ import annotations

from pathlib import Path

import joblib
import numpy
import sklearn


ROOT = Path(__file__).resolve().parents[1]


def test_f6_runtime_dependencies_are_pinned_to_verified_versions() -> None:
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "numpy==2.5.1" in requirements
    assert "scikit-learn==1.5.2" in requirements
    assert "joblib==1.4.2" in requirements
    assert numpy.__version__ == "2.5.1"
    assert sklearn.__version__ == "1.5.2"
    assert joblib.__version__ == "1.4.2"
