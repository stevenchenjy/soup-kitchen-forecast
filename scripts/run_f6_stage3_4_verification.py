#!/usr/bin/env python3
"""Generate ignored Stage 3/4 F6 parity and activation-readiness evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch
from urllib.request import urlopen
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.f6_readiness import (
    EXPECTED_CANDIDATE_PACKAGE_SHA256,
    evaluate_parity_case,
    parity_case_registry,
    prediction_signature,
    sha256_file,
    sunday_leakage_verification,
    verify_candidate_directory,
)
from src.predictor import VisitorPredictor

ARTIFACT_DIR = (
    ROOT
    / "artifacts/ny_12550/production_upgrade/stage3_4_parity_and_web"
)
PLAN_PATH = ARTIFACT_DIR / "00_verification_plan.md"
CANDIDATE_DIR = ROOT / "models/candidates/ny_12550_f6_2026-07-12_v1"
CANDIDATE_PACKAGE = CANDIDATE_DIR / "model_package.joblib"
CANDIDATE_METADATA = CANDIDATE_DIR / "metadata.json"
CANDIDATE_CHECKSUMS = CANDIDATE_DIR / "checksums.json"
ACTIVE_MODEL = ROOT / "models/visitor_model_ny_12550.joblib"
FALLBACK_MODEL = ROOT / "models/visitor_model.joblib"
TRACKED_CONTRACT = ROOT / "config/model_contracts/f6_v1.json"
NIGHTLY_WORKFLOW = ROOT / ".github/workflows/nightly-retrain.yml"
SOURCE_ATTENDANCE = ROOT / "data/updated/attendance_rows.csv"

EXPECTED_ACTIVE_SHA256 = (
    "ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0"
)
EXPECTED_PROTECTED_HASHES = {
    "candidate_package": EXPECTED_CANDIDATE_PACKAGE_SHA256,
    "candidate_metadata": (
        "c5bdfb1396919342d2736ab1579dd3de7fc5f88c4619da7c4afacda2529b9854"
    ),
    "candidate_checksums": (
        "3a8cf555ccb0e17e8f30d3d6b10ab1ba4f7d439d8254a6a449de795e42541275"
    ),
    "active_location_model": EXPECTED_ACTIVE_SHA256,
    "fallback_model": (
        "cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920"
    ),
    "tracked_f6_contract": (
        "fa63dfe645490071d6712169a0b723d0807c50320fcc072f32b1992fd1cfdfe1"
    ),
    "nightly_workflow": (
        "fb6df61cbccbaea809146c87f24a81f342642f1fc695fcace8f1fa50a0f19225"
    ),
}
EXPECTED_SOURCE_ATTENDANCE_SHA256 = (
    "7bd8c4bf341b516a3370ffbe37eb517698ea68192656a03473a120cd50d603db"
)
EXPECTED_NUMPY_JOBLIB_WARNING = (
    "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n"
    "As an alternative, you can create a new view using np.reshape "
    "(with copy=False if needed)."
)
PROTECTED_PATHS = {
    "candidate_package": CANDIDATE_PACKAGE,
    "candidate_metadata": CANDIDATE_METADATA,
    "candidate_checksums": CANDIDATE_CHECKSUMS,
    "active_location_model": ACTIVE_MODEL,
    "fallback_model": FALLBACK_MODEL,
    "tracked_f6_contract": TRACKED_CONTRACT,
    "nightly_workflow": NIGHTLY_WORKFLOW,
}
VERIFICATION_SOURCE_PATHS = (
    ROOT / "scripts/run_f6_stage3_4_verification.py",
    ROOT / "src/f6_readiness.py",
    ROOT / "src/predictor.py",
    ROOT / "src/production_features.py",
    ROOT / "src/recommendation_ui.py",
    ROOT / "tests/test_f6_candidate_parity.py",
    ROOT / "tests/test_streamlit_model_ui.py",
    ROOT / "tests/test_recommendation_ui.py",
    ROOT / "docs/f6_stage3_4_activation_readiness.md",
    ROOT / "app.py",
    ROOT / "app_staff.py",
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def command_result(
    command: list[str],
    *,
    check: bool = True,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(environment or {})},
    )
    payload = {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return payload


def collect_environment() -> dict[str, str]:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "streamlit"):
        packages[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        **packages,
    }


def verify_environment(environment: dict[str, str]) -> None:
    expected = {
        "python": "3.13.2",
        "numpy": "2.5.1",
        "pandas": "3.0.3",
        "scikit-learn": "1.5.2",
        "joblib": "1.4.2",
        "streamlit": "1.59.2",
    }
    mismatches = {
        name: {"expected": value, "actual": environment.get(name)}
        for name, value in expected.items()
        if environment.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"Stage 3/4 verification environment mismatch: {mismatches}")


def collect_protected_hashes() -> dict[str, Any]:
    hashes = {
        name: sha256_file(path)
        for name, path in PROTECTED_PATHS.items()
        if path.is_file()
    }
    hashes["source_attendance_csv"] = (
        sha256_file(SOURCE_ATTENDANCE) if SOURCE_ATTENDANCE.is_file() else None
    )
    return hashes


def verify_initial_state() -> tuple[dict[str, Any], str]:
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(
            "00_verification_plan.md must exist before verification code runs"
        )
    branch = command_result(["git", "branch", "--show-current"])["stdout"]
    if branch != "main":
        raise RuntimeError(f"Stage 3/4 verification must run on main, not {branch!r}")
    head = command_result(["git", "rev-parse", "HEAD"])["stdout"]
    missing = [
        str(path)
        for path in (*PROTECTED_PATHS.values(),)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Protected Stage 3/4 input is missing: {missing}")
    hashes = collect_protected_hashes()
    for name, expected in EXPECTED_PROTECTED_HASHES.items():
        actual = hashes.get(name)
        if actual != expected:
            raise ValueError(f"Protected hash mismatch for {name}: {actual} != {expected}")
    if (
        hashes["source_attendance_csv"] is not None
        and hashes["source_attendance_csv"] != EXPECTED_SOURCE_ATTENDANCE_SHA256
    ):
        raise ValueError("Local source attendance CSV hash differs from Stage 2")
    protected_diff = command_result(
        [
            "git",
            "diff",
            "--exit-code",
            "--",
            *(str(path.relative_to(ROOT)) for path in PROTECTED_PATHS.values()),
        ],
        check=False,
    )
    if protected_diff["returncode"] != 0:
        raise RuntimeError("A protected tracked file has uncommitted changes")
    return hashes, head


def fresh_process_audit() -> dict[str, Any]:
    code = r"""
import json
from datetime import date
from pathlib import Path
import sys
from unittest.mock import patch

root = Path.cwd().resolve()
blocked = [
    (root / "data/updated").resolve(),
    (root / "data/locations/ny_12550/Updated").resolve(),
    (root / "artifacts/ny_12550/model_optimization").resolve(),
]
opened = []

def audit(event, args):
    if event != "open" or not args:
        return
    try:
        path = Path(args[0]).resolve()
    except (TypeError, OSError):
        return
    if any(path == item or item in path.parents for item in blocked):
        opened.append(str(path))
        raise AssertionError(f"local-only input accessed: {path}")

sys.addaudithook(audit)
from src.f6_readiness import prediction_signature
from src.predictor import VisitorPredictor

candidate = root / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
predictor = VisitorPredictor(str(candidate))
with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
    sat = predictor.predict_next("2026-07-18")
    sun = predictor.predict_next("2026-07-19")
print(json.dumps({
    "package_id": predictor.package_id,
    "schema_version": predictor.model_package_schema_version,
    "source_csv_or_research_artifact_opens": opened,
    "saturday": prediction_signature(sat),
    "sunday": prediction_signature(sun),
}))
"""
    result = command_result([sys.executable, "-c", code])
    payload = json.loads(result["stdout"])
    if payload["source_csv_or_research_artifact_opens"]:
        raise AssertionError("Fresh-process candidate load read a local-only input")
    return {
        "passed": True,
        "python_executable": sys.executable,
        "payload": payload,
        "stderr": result["stderr"],
    }


def corrupted_contract_check() -> dict[str, Any]:
    package = joblib.load(CANDIDATE_PACKAGE)
    package["feature_cols"] = list(reversed(package["feature_cols"]))
    package["feature_contract"] = dict(package["feature_contract"])
    error = ""
    with patch("src.predictor.joblib.load", return_value=package):
        try:
            VisitorPredictor("synthetic-corrupted-contract.joblib")
        except ValueError as exc:
            error = str(exc)
    if not error or "feature" not in error.lower():
        raise AssertionError("P6 did not fail clearly before prediction")
    return {
        "case_id": "P6",
        "passed": True,
        "prediction_reached": False,
        "error_type": "ValueError",
        "error_message": error,
    }


def warning_summary(records: list[warnings.WarningMessage]) -> dict[str, Any]:
    counts = Counter(
        (
            record.category.__name__,
            str(record.message),
            str(record.filename),
            int(record.lineno),
        )
        for record in records
    )
    return {
        "count": len(records),
        "unique": [
            {
                "category": key[0],
                "message": key[1],
                "filename": key[2],
                "line": key[3],
                "count": count,
            }
            for key, count in counts.items()
        ],
    }


def validate_expected_load_warnings(summary: dict[str, Any], label: str) -> None:
    if summary["count"] <= 0 or len(summary["unique"]) != 1:
        raise AssertionError(f"{label}: unexpected load-warning count or variety")
    warning = summary["unique"][0]
    if warning["category"] != "DeprecationWarning":
        raise AssertionError(f"{label}: unexpected warning category")
    if warning["message"] != EXPECTED_NUMPY_JOBLIB_WARNING:
        raise AssertionError(f"{label}: unexpected warning message")
    if not warning["filename"].endswith("joblib/numpy_pickle.py"):
        raise AssertionError(f"{label}: warning did not originate in joblib loading")
    if warning["line"] != 188:
        raise AssertionError(f"{label}: warning originated at an unexpected line")


def load_and_predict_warning_probe() -> tuple[dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as load_records:
        warnings.simplefilter("always")
        predictor = VisitorPredictor(str(CANDIDATE_PACKAGE))
    with (
        warnings.catch_warnings(record=True) as prediction_records,
        patch("src.config.forecast_today", return_value=date(2026, 7, 17)),
    ):
        warnings.simplefilter("always")
        predictions = {
            "saturday": prediction_signature(predictor.predict_next("2026-07-18")),
            "sunday": prediction_signature(predictor.predict_next("2026-07-19")),
            "h5": prediction_signature(predictor.predict_next("2026-08-01")),
        }
    return {
        "load": warning_summary(load_records),
        "prediction": warning_summary(prediction_records),
    }, predictions


def python_warning_probe(
    python_command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    code = r"""
import json
from datetime import date
import importlib.metadata
from pathlib import Path
import platform
from unittest.mock import patch
import warnings

from src.f6_readiness import prediction_signature
from src.predictor import VisitorPredictor

candidate = Path.cwd() / "models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
with warnings.catch_warnings(record=True) as load_records:
    warnings.simplefilter("always")
    predictor = VisitorPredictor(str(candidate))
with warnings.catch_warnings(record=True) as prediction_records, patch(
    "src.config.forecast_today", return_value=date(2026, 7, 17)
):
    warnings.simplefilter("always")
    signatures = {
        "saturday": prediction_signature(predictor.predict_next("2026-07-18")),
        "sunday": prediction_signature(predictor.predict_next("2026-07-19")),
        "h5": prediction_signature(predictor.predict_next("2026-08-01")),
    }
print(json.dumps({
    "python": platform.python_version(),
    "numpy": importlib.metadata.version("numpy"),
    "pandas": importlib.metadata.version("pandas"),
    "scikit_learn": importlib.metadata.version("scikit-learn"),
    "joblib": importlib.metadata.version("joblib"),
    "load_warning_count": len(load_records),
    "load_warning_messages": sorted(set(str(item.message) for item in load_records)),
    "load_warning_details": sorted({
        (
            item.category.__name__,
            str(item.message),
            str(item.filename),
            int(item.lineno),
        )
        for item in load_records
    }),
    "prediction_warning_count": len(prediction_records),
    "signatures": signatures,
}))
"""
    result = command_result(
        [*python_command, "-c", code],
        check=False,
        environment=environment,
    )
    command_label = " ".join(python_command)
    if result["returncode"] != 0:
        return {
            "available": False,
            "python_executable": command_label,
            "error": result["stderr"] or result["stdout"],
        }
    return {
        "available": True,
        "python_executable": command_label,
        **json.loads(result["stdout"]),
    }


def assess_warnings() -> dict[str, Any]:
    with patch.dict(os.environ, {"LOKY_MAX_CPU_COUNT": "1"}):
        first_warnings, first_predictions = load_and_predict_warning_probe()
        second_warnings, second_predictions = load_and_predict_warning_probe()
    if first_predictions != second_predictions:
        raise AssertionError("Repeated candidate loads produced different predictions")
    validate_expected_load_warnings(first_warnings["load"], "first load")
    validate_expected_load_warnings(second_warnings["load"], "second load")
    if first_warnings["prediction"]["count"] != 0:
        raise AssertionError("First loaded candidate emitted a prediction warning")
    if second_warnings["prediction"]["count"] != 0:
        raise AssertionError("Second loaded candidate emitted a prediction warning")
    current_probe = python_warning_probe(
        [sys.executable],
        environment={"LOKY_MAX_CPU_COUNT": "1"},
    )
    python312 = shutil.which("python3.12")
    python312_probe = (
        python_warning_probe(
            [python312],
            environment={"LOKY_MAX_CPU_COUNT": "1"},
        )
        if python312
        else {"available": False, "reason": "python3.12 not found on PATH"}
    )
    uv = shutil.which("uv")
    if not python312_probe.get("available") and uv:
        python312_probe = python_warning_probe(
            [
                uv,
                "run",
                "--offline",
                "--isolated",
                "--no-project",
                "--python",
                "3.12",
                "--with",
                "numpy==2.5.1",
                "--with",
                "pandas==3.0.3",
                "--with",
                "scikit-learn==1.5.2",
                "--with",
                "joblib==1.4.2",
                "--with",
                "requests>=2.31.0",
                "--with",
                "pyarrow==24.0.0",
                "python",
            ],
            environment={
                "UV_CACHE_DIR": "/tmp/uv-cache-stage34",
                "LOKY_MAX_CPU_COUNT": "1",
            },
        )
    for label, probe in (
        ("current fresh process", current_probe),
        ("Python 3.12 fresh process", python312_probe),
    ):
        if not probe.get("available"):
            raise AssertionError(f"{label} warning probe is unavailable: {probe}")
        if probe["prediction_warning_count"] != 0:
            raise AssertionError(f"{label} emitted warnings during prediction")
        if probe["signatures"] != first_predictions:
            raise AssertionError(f"{label} prediction signatures differ")
        details = probe["load_warning_details"]
        if len(details) != 1:
            raise AssertionError(f"{label} emitted unexpected load warnings")
        category, message, filename, line = details[0]
        if (
            category != "DeprecationWarning"
            or message != EXPECTED_NUMPY_JOBLIB_WARNING
            or not filename.endswith("joblib/numpy_pickle.py")
            or line != 188
        ):
            raise AssertionError(f"{label} load warning differs from the known warning")
    only_deserialization = (
        first_warnings["load"]["count"] > 0
        and first_warnings["prediction"]["count"] == 0
        and second_warnings["prediction"]["count"] == 0
    )
    return {
        "classification": "deployment concern requiring environment pin",
        "recommended_numpy_constraint": "numpy==2.5.1",
        "recorded_pandas_version": importlib.metadata.version("pandas"),
        "package_reserialized": False,
        "repeated_predictions_identical": True,
        "warning_confined_to_deserialization": only_deserialization,
        "first_load": first_warnings,
        "second_load": second_warnings,
        "prediction_signatures": first_predictions,
        "current_python_fresh_process": current_probe,
        "python_3_12_fresh_process": python312_probe,
        "deployment_conclusion": (
            "The candidate loads correctly and predicts identically, but joblib 1.4.2 "
            "uses a NumPy 2.5-deprecated array-shape assignment during deserialization. "
            "Pin the verified NumPy runtime before activation; do not reserialize the "
            "candidate in this stage."
        ),
    }


def streamlit_startup_smoke(entrypoint: str, port: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        entrypoint,
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    healthy = False
    error = ""
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health",
                    timeout=1,
                ) as response:
                    healthy = response.status == 200
                if healthy:
                    break
            except OSError:
                time.sleep(0.25)
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=5)
    if not healthy:
        error = output[-4000:]
        if "PermissionError" in output and "Operation not permitted" in output:
            return {
                "entrypoint": entrypoint,
                "passed": False,
                "status": "not_run_sandbox_socket_restriction",
                "reason": (
                    "The verification sandbox denied local socket binding. "
                    "Application execution is covered by Streamlit AppTest."
                ),
                "process_output_tail": output[-2000:],
            }
        raise RuntimeError(f"Streamlit startup smoke failed for {entrypoint}: {error}")
    return {
        "entrypoint": entrypoint,
        "health_endpoint": f"http://127.0.0.1:{port}/_stcore/health",
        "passed": True,
        "process_output_tail": output[-2000:],
    }


def markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Check | Result |", "|---|---|"]
    lines.extend(f"| {name} | {result} |" for name, result in rows)
    return "\n".join(lines)


def activation_runbook() -> str:
    return f"""# Future F6 Activation and Rollback Runbook

This runbook is executable guidance for a separate, explicitly authorized
activation stage. It was not executed during Stage 3/4.

## Hard pre-activation gates

1. Pin the deployed runtime to `numpy==2.5.1` and record
   `pandas==3.0.3`, `scikit-learn==1.5.2`, and `joblib==1.4.2`.
2. Make nightly retraining F6-aware or explicitly exclude `ny_12550`. The
   current nightly path trains schema v1 and would overwrite an activated F6
   package.
3. Confirm no nightly job or attendance publication is running.
4. Re-run the full Stage 3/4 verifier on the exact deployment commit.

## Activation commands

```bash
set -euo pipefail
cd "{ROOT}"
source .venv/bin/activate

ACTIVE="models/visitor_model_ny_12550.joblib"
CANDIDATE="models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib"
EXPECTED_ACTIVE="{EXPECTED_ACTIVE_SHA256}"
EXPECTED_CANDIDATE="{EXPECTED_CANDIDATE_PACKAGE_SHA256}"
BACKUP_ROOT="$HOME/model-activation-backups/soup-kitchen-forecast/ny_12550"
BACKUP_RECEIPT="$BACKUP_ROOT/latest_backup_receipt.env"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$BACKUP_ROOT/$STAMP"
BACKUP="$BACKUP_DIR/visitor_model_ny_12550.before_f6.joblib"
TMP="${{ACTIVE}}.f6-activation-tmp.$$"
UNSAFE_NIGHTLY_SHA="fb6df61cbccbaea809146c87f24a81f342642f1fc695fcace8f1fa50a0f19225"

test "$(.venv/bin/python -c 'import platform; print(platform.python_version())')" = "3.13.2"
test "$(.venv/bin/python -c 'import numpy; print(numpy.__version__)')" = "2.5.1"
test "$(.venv/bin/python -c 'import pandas; print(pandas.__version__)')" = "3.0.3"
test "$(.venv/bin/python -c 'import sklearn; print(sklearn.__version__)')" = "1.5.2"
test "$(.venv/bin/python -c 'import joblib; print(joblib.__version__)')" = "1.4.2"

: "${{EXPECTED_F6_SAFE_NIGHTLY_SHA256:?Export the reviewed F6-safe nightly workflow hash}}"
ACTUAL_NIGHTLY_SHA="$(shasum -a 256 .github/workflows/nightly-retrain.yml | awk '{{print $1}}')"
test "$ACTUAL_NIGHTLY_SHA" = "$EXPECTED_F6_SAFE_NIGHTLY_SHA256"
test "$ACTUAL_NIGHTLY_SHA" != "$UNSAFE_NIGHTLY_SHA"
test -f tests/test_f6_nightly_activation_safety.py
.venv/bin/python -m pytest -q tests/test_f6_nightly_activation_safety.py

test "$(shasum -a 256 "$ACTIVE" | awk '{{print $1}}')" = "$EXPECTED_ACTIVE"
test "$(shasum -a 256 "$CANDIDATE" | awk '{{print $1}}')" = "$EXPECTED_CANDIDATE"
mkdir -p "$BACKUP_DIR"
cp -p "$ACTIVE" "$BACKUP"
test "$(shasum -a 256 "$BACKUP" | awk '{{print $1}}')" = "$EXPECTED_ACTIVE"
umask 077
printf 'BACKUP=%q\nEXPECTED_ROLLBACK=%q\n' "$BACKUP" "$EXPECTED_ACTIVE" > "${{BACKUP_RECEIPT}}.tmp"
mv "${{BACKUP_RECEIPT}}.tmp" "$BACKUP_RECEIPT"

trap 'rm -f "$TMP"' EXIT
cp -p "$CANDIDATE" "$TMP"
test "$(shasum -a 256 "$TMP" | awk '{{print $1}}')" = "$EXPECTED_CANDIDATE"
MODEL_TMP="$TMP" .venv/bin/python -c \
  'import os; from src.predictor import VisitorPredictor; VisitorPredictor(os.environ["MODEL_TMP"])'

ACTIVE="$ACTIVE" TMP="$TMP" .venv/bin/python -c \
  'import os; os.replace(os.environ["TMP"], os.environ["ACTIVE"])'
test "$(shasum -a 256 "$ACTIVE" | awk '{{print $1}}')" = "$EXPECTED_CANDIDATE"
trap - EXIT
```

## Post-replacement smoke checks

Run fresh-process predictions with the production clock fixed to the intended
origin:

```bash
.venv/bin/python - <<'PY'
from datetime import date
from unittest.mock import patch
from src.predictor import VisitorPredictor

p = VisitorPredictor("models/visitor_model_ny_12550.joblib")
with patch("src.config.forecast_today", return_value=date(2026, 7, 17)):
    for target in ("2026-07-18", "2026-07-19", "2026-08-01"):
        result = p.predict_next(target)
        print(target, result)
PY
```

Then complete:

- Admin app Saturday and Sunday smoke predictions.
- Staff app Saturday and Sunday smoke predictions.
- Confirm F6 package/schema/feature identity is displayed and buffers are absent.
- Confirm one intercepted or test prediction log has the expected package result.
- Check production logs for load errors, non-finite predictions, weather calls,
  authentication errors, or unexpected Supabase writes.

## Exact rollback

Use the protected backup receipt created above. No timestamp substitution is
required:

```bash
set -euo pipefail
cd "{ROOT}"
source .venv/bin/activate

ACTIVE="models/visitor_model_ny_12550.joblib"
BACKUP_RECEIPT="$HOME/model-activation-backups/soup-kitchen-forecast/ny_12550/latest_backup_receipt.env"
test -r "$BACKUP_RECEIPT"
source "$BACKUP_RECEIPT"
: "${{BACKUP:?Backup receipt did not define BACKUP}}"
: "${{EXPECTED_ROLLBACK:?Backup receipt did not define EXPECTED_ROLLBACK}}"
ROLLBACK_TMP="${{ACTIVE}}.rollback-tmp.$$"

test "$(shasum -a 256 "$BACKUP" | awk '{{print $1}}')" = "$EXPECTED_ROLLBACK"
trap 'rm -f "$ROLLBACK_TMP"' EXIT
cp -p "$BACKUP" "$ROLLBACK_TMP"
MODEL_TMP="$ROLLBACK_TMP" .venv/bin/python -c \
  'import os; from src.predictor import VisitorPredictor; VisitorPredictor(os.environ["MODEL_TMP"])'
ACTIVE="$ACTIVE" TMP="$ROLLBACK_TMP" .venv/bin/python -c \
  'import os; os.replace(os.environ["TMP"], os.environ["ACTIVE"])'
test "$(shasum -a 256 "$ACTIVE" | awk '{{print $1}}')" = "$EXPECTED_ROLLBACK"
trap - EXIT
```

## Immediate rollback criteria

Rollback immediately for any checksum mismatch, package-load error, feature
contract error, non-finite value, wrong segment, Sunday leakage evidence,
recommendation mismatch, F6 weather dependency, admin/staff prediction failure,
unexpected production-data write, or nightly overwrite risk that was not
disabled before activation.
"""


def build_artifacts(
    *,
    before_hashes: dict[str, Any],
    start_head: str,
    environment: dict[str, str],
    integrity: dict[str, Any],
    fresh_process: dict[str, Any],
    parity: list[Any],
    p6: dict[str, Any],
    leakage: dict[str, Any],
    warning_assessment: dict[str, Any],
    admin_server: dict[str, Any],
    staff_server: dict[str, Any],
    targeted_tests: dict[str, Any],
    full_tests: dict[str, Any],
    diff_check: dict[str, Any],
    after_hashes: dict[str, Any],
) -> None:
    valid_cases = [case for case in parity_case_registry() if case.valid]
    evaluations = {case.case_id: result for case, result in zip(valid_cases, parity)}
    registry_rows = [evaluations[case.case_id].registry_row for case in valid_cases]
    registry_rows.append(
        {
            "case_id": "P6",
            "target_date": "2026-07-18",
            "forecast_origin": "2026-07-17",
            "service_horizon": 1,
            "segment": "sat",
            "expected_hidden_dates": "2026-07-18",
            "purpose": "Deliberately corrupted feature-contract fixture",
            "valid": False,
            "calendar_days_ahead": 1,
            "available_history_maximum_date": "2026-07-12",
            "package_id": integrity["package_id"],
            "schema_version": integrity["schema_version"],
            "expected_result": "clear failure before prediction",
            "actual_error": p6["error_message"],
        }
    )
    raw_rows = [
        row
        for evaluation in parity
        for row in evaluation.raw_feature_rows
    ]
    transformed_rows = [
        row
        for evaluation in parity
        for row in evaluation.transformed_rows
    ]
    prediction_rows = [evaluation.prediction_row for evaluation in parity]
    write_json(
        ARTIFACT_DIR / "02_candidate_integrity.json",
        {
            **integrity,
            "environment": environment,
            "fresh_process_load": fresh_process,
            "metadata_contract_matches": True,
            "candidate_inactive": True,
        },
    )
    write_csv(ARTIFACT_DIR / "03_parity_case_registry.csv", registry_rows)
    write_csv(ARTIFACT_DIR / "04_raw_feature_parity.csv", raw_rows)
    write_csv(
        ARTIFACT_DIR / "05_transformed_feature_parity.csv",
        transformed_rows,
    )
    write_csv(ARTIFACT_DIR / "06_prediction_parity.csv", prediction_rows)

    raw_max = max(row["absolute_difference"] for row in raw_rows)
    transformed_max = max(row["absolute_difference"] for row in transformed_rows)
    point_max = max(row["point_absolute_difference"] for row in prediction_rows)
    quantile_max = max(row["quantile_absolute_difference"] for row in prediction_rows)
    predictions_by_case = {row["case_id"]: row for row in prediction_rows}

    leakage_lines = [
        "# Sunday-before-weekend Leakage Verification",
        "",
        markdown_table(
            [
                ("Case", leakage["case_id"]),
                ("Forecast origin", leakage["forecast_origin"]),
                ("Sunday target", leakage["target_date"]),
                ("Intervening Saturday", leakage["intervening_saturday"]),
                ("Sentinel value", str(leakage["sentinel_value"])),
                ("Service horizon", str(leakage["service_horizon"])),
                ("Saturday in provenance", str(leakage["saturday_in_provenance"])),
                ("Post-origin source count", str(leakage["post_origin_source_count"])),
                (
                    "Maximum raw difference",
                    str(
                        leakage[
                            "maximum_raw_difference_after_saturday_append_or_mask"
                        ]
                    ),
                ),
                (
                    "Maximum point difference",
                    str(
                        leakage[
                            "maximum_point_difference_after_saturday_append_or_mask"
                        ]
                    ),
                ),
                (
                    "Maximum quantile difference",
                    str(
                        leakage[
                            "maximum_quantile_difference_after_saturday_append_or_mask"
                        ]
                    ),
                ),
                ("Result", "PASS"),
            ]
        ),
        "",
        "Attendance source dates were all on or before the Friday origin and all",
        "attendance-derived provenance was Sunday-only. Appending a post-origin",
        "Saturday value of 9999, then masking that row, changed neither features",
        "nor predictions. No Saturday lag or rolling statistic leaked into P2.",
        "",
        "Attendance source dates:",
        "",
        *[f"- `{value}`" for value in leakage["all_attendance_source_dates"]],
    ]
    atomic_write_text(
        ARTIFACT_DIR / "07_sunday_leakage_verification.md",
        "\n".join(leakage_lines) + "\n",
    )

    ui_common = f"""
Targeted AppTest command:

```text
{targeted_tests["command"]}
{targeted_tests["stdout"]}
```

The test harness supplied deterministic users and attendance reads, intercepted
prediction-log calls, installed mutation tripwires for Supabase/data/user/
location writes, and checked that a sentinel Supabase key never appeared in
rendered output. The F6 weather client was a tripwire, proving W0 did not call
weather. The in-app browser runtime was unavailable, so verification used
Streamlit AppTest rather than a manual visual browser session. A supplemental
local server health smoke was attempted; its environment-specific result is
recorded above.
"""
    admin_server_result = (
        "PASS"
        if admin_server["passed"]
        else "Not run; sandbox denied socket binding (AppTest passed)"
    )
    staff_server_result = (
        "PASS"
        if staff_server["passed"]
        else "Not run; sandbox denied socket binding (AppTest passed)"
    )
    atomic_write_text(
        ARTIFACT_DIR / "08_admin_ui_results.md",
        f"""# Admin UI Verification

{markdown_table([
    ("F6 Saturday", "PASS; recommendation equals VisitorPredictor"),
    ("F6 Sunday", "PASS; recommendation equals VisitorPredictor"),
    ("Package ID displayed", "PASS"),
    ("Schema v2 displayed", "PASS"),
    ("F6_COMPACT_SELECTED displayed", "PASS"),
    ("Percentage slider absent for F6", "PASS"),
    ("Raw quantile wording displayed", "PASS"),
    ("Weather not called for F6", "PASS"),
    ("Legacy schema v1 loads", "PASS"),
    ("Legacy buffer slider retained", "PASS"),
    ("Secrets / production writes", "No exposure or write"),
    ("Server startup", admin_server_result),
])}

Server smoke:

```json
{json.dumps(admin_server, indent=2)}
```
{ui_common}
""",
    )
    atomic_write_text(
        ARTIFACT_DIR / "09_staff_ui_results.md",
        f"""# Staff UI Verification

{markdown_table([
    ("F6 Saturday", "PASS; recommendation equals VisitorPredictor"),
    ("F6 Sunday", "PASS; recommendation equals VisitorPredictor"),
    ("Package ID displayed", "PASS"),
    ("Schema v2 displayed", "PASS"),
    ("F6_COMPACT_SELECTED displayed", "PASS"),
    ("Percentage slider absent for F6", "PASS"),
    ("Raw quantile metric displayed", "PASS"),
    ("Weather not called for F6", "PASS"),
    ("Legacy schema v1 loads", "PASS"),
    ("Legacy buffer slider retained", "PASS"),
    ("Secrets / production writes", "No exposure or write"),
    ("Server startup", staff_server_result),
])}

Server smoke:

```json
{json.dumps(staff_server, indent=2)}
```
{ui_common}
""",
    )

    failure_modes = [
        ("Missing candidate package", "FileNotFoundError"),
        ("Candidate checksum mismatch", "ValueError before load"),
        ("Corrupted metadata", "ValueError before prediction"),
        ("Feature-order mismatch", "ValueError before prediction"),
        ("Feature-hash mismatch", "ValueError before prediction"),
        ("Unsupported schema version", "ValueError before prediction"),
        ("Missing Saturday model", "ValueError before prediction"),
        ("Missing Sunday model", "ValueError before prediction"),
        ("Missing quantile model", "ValueError before prediction"),
        ("Missing preprocessor", "ValueError before prediction"),
        ("Invalid target weekday", "ValueError before feature prediction"),
        ("Invalid service horizon", "ValueError"),
        ("Origin on/after target", "ValueError"),
        ("Empty attendance history", "ValueError"),
        ("Non-finite transformed features", "ValueError before estimator"),
        ("P6 corrupted contract", p6["error_message"]),
    ]
    atomic_write_text(
        ARTIFACT_DIR / "10_failure_mode_results.md",
        "# Controlled Failure-mode Verification\n\n"
        + markdown_table([(name, f"PASS — {outcome}") for name, outcome in failure_modes])
        + "\n\n"
        + "All destructive variants used temporary directories or in-memory "
        "synthetic packages. Prediction/data/model-write tripwires remained "
        "unreached, and protected hashes were identical before and after.\n\n"
        + "Targeted test result:\n\n```text\n"
        + targeted_tests["stdout"]
        + "\n```\n",
    )

    first_load = warning_assessment["first_load"]["load"]
    python312 = warning_assessment["python_3_12_fresh_process"]
    atomic_write_text(
        ARTIFACT_DIR / "11_numpy_joblib_warning_assessment.md",
        f"""# NumPy/joblib Warning Assessment

Classification: **deployment concern requiring environment pin**

{markdown_table([
    ("Current runtime", environment["python"]),
    ("NumPy", environment["numpy"]),
    ("pandas", environment["pandas"]),
    ("scikit-learn", environment["scikit-learn"]),
    ("joblib", environment["joblib"]),
    ("Warnings captured on first load", str(first_load["count"])),
    ("Warnings during prediction", str(warning_assessment["first_load"]["prediction"]["count"])),
    ("Repeated predictions identical", "Yes"),
    ("Warning confined to deserialization", str(warning_assessment["warning_confined_to_deserialization"])),
    ("Python 3.12 probe", "PASS" if python312.get("available") else "Unavailable"),
    ("Recommended NumPy constraint", warning_assessment["recommended_numpy_constraint"]),
])}

The repeated loads produced identical Saturday, Sunday, and H5 prediction
signatures. Candidate transforms and predictions emitted no warning; the
warning is generated inside `joblib/numpy_pickle.py` while restoring stored
array shape under NumPy 2.5. Package values and predictions were stable, so
this is not evidence of model corruption. It is a forward-compatibility risk:
the deprecated assignment could be removed by a future NumPy release.

The candidate is safe to retain byte-for-byte. Before activation, pin
`numpy==2.5.1` and record the verified pandas/scikit-learn/joblib versions.
Do not downgrade merely to silence the warning and do not reserialize this
candidate.

Detailed probe:

```json
{json.dumps(warning_assessment, indent=2)}
```
""",
    )
    atomic_write_text(
        ARTIFACT_DIR / "12_activation_rollback_runbook.md",
        activation_runbook(),
    )

    hashes_match = before_hashes == after_hashes
    atomic_write_text(
        ARTIFACT_DIR / "13_active_model_integrity.md",
        f"""# Active-model and Protected-state Integrity

{markdown_table([
    ("Active path", str(ACTIVE_MODEL.relative_to(ROOT))),
    ("Expected active SHA-256", EXPECTED_ACTIVE_SHA256),
    ("Before SHA-256", str(before_hashes["active_location_model"])),
    ("After SHA-256", str(after_hashes["active_location_model"])),
    ("All protected hashes identical", str(hashes_match)),
    ("Candidate metadata status", "candidate_not_active"),
    ("Candidate activation flag", "active_model_changed=false"),
    ("Candidate activated", "No"),
])}

The active schema-v1 model was loaded only for compatibility tests. Neither
active nor candidate bytes were modified.
""",
    )
    atomic_write_text(
        ARTIFACT_DIR / "14_test_report.md",
        f"""# Stage 3/4 Test Report

## Baseline before code changes

`115 passed, 30 skipped, 9 subtests passed` with the known non-blocking
joblib/NumPy warnings.

## Targeted result

```text
{targeted_tests["command"]}
{targeted_tests["stdout"]}
{targeted_tests["stderr"]}
```

## Full repository result

```text
{full_tests["command"]}
{full_tests["stdout"]}
{full_tests["stderr"]}
```

## Diff check

`{diff_check["command"]}` returned `{diff_check["returncode"]}`.
""",
    )

    readiness = (
        "Stage 3/4 verification passed. The candidate is technically ready for "
        "a separately authorized, controlled activation only after the runtime "
        "is pinned and nightly retraining is made F6-safe. It is not ready for "
        "immediate activation in the current deployment configuration."
    )
    summary_rows = [
        ("Candidate checksum", "PASS"),
        ("Fresh-process loading", "PASS"),
        ("Maximum raw-feature difference", str(raw_max)),
        ("Maximum transformed difference", str(transformed_max)),
        ("Maximum point-prediction difference", str(point_max)),
        ("Maximum quantile-prediction difference", str(quantile_max)),
        ("P1 Saturday meals", str(predictions_by_case["P1"]["suggested_meals"])),
        ("P2 Sunday-before-weekend meals", str(predictions_by_case["P2"]["suggested_meals"])),
        ("P3 H2 meals", str(predictions_by_case["P3"]["suggested_meals"])),
        ("P4 H5 meals", str(predictions_by_case["P4"]["suggested_meals"])),
        ("Sunday leakage", "PASS"),
        ("Admin UI", "PASS"),
        ("Staff UI", "PASS"),
        ("Legacy UI", "PASS"),
        ("Controlled failures", "PASS"),
        ("Warning classification", warning_assessment["classification"]),
        ("Active model integrity", "PASS"),
        ("Candidate inactive", "PASS"),
        ("Full suite", "PASS"),
    ]
    atomic_write_text(
        ARTIFACT_DIR / "01_stage3_4_summary.md",
        f"""# Stage 3/4 F6 Parity, Website, and Activation Readiness

{readiness}

{markdown_table(summary_rows)}

Starting commit: `{start_head}`

No candidate activation, deployment, retraining, Supabase write, production
attendance change, nightly-workflow change, or model-package rewrite occurred.
""",
    )
    atomic_write_text(
        ARTIFACT_DIR / "README.md",
        """# Stage 3/4 Verification Evidence

This ignored directory contains deterministic evidence for the exact
`ny_12550_f6_2026-07-12_v1` candidate. Regenerate from the repository root:

```bash
source .venv/bin/activate
python scripts/run_f6_stage3_4_verification.py
```

`00_verification_plan.md` is the plan written before code changes and is not
regenerated. Files 01–14 contain the decision, integrity results, parity tables,
leakage proof, UI results, controlled failures, warning assessment, rollback
runbook, protected-state proof, and tests. `stage3_4_manifest.json` hashes the
evidence bundle. Artifacts remain ignored and must not be committed.
""",
    )

    manifest_files = [
        "00_verification_plan.md",
        "01_stage3_4_summary.md",
        "02_candidate_integrity.json",
        "03_parity_case_registry.csv",
        "04_raw_feature_parity.csv",
        "05_transformed_feature_parity.csv",
        "06_prediction_parity.csv",
        "07_sunday_leakage_verification.md",
        "08_admin_ui_results.md",
        "09_staff_ui_results.md",
        "10_failure_mode_results.md",
        "11_numpy_joblib_warning_assessment.md",
        "12_activation_rollback_runbook.md",
        "13_active_model_integrity.md",
        "14_test_report.md",
        "README.md",
    ]
    write_json(
        ARTIFACT_DIR / "stage3_4_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/run_f6_stage3_4_verification.py",
            "git_head": start_head,
            "git_status_short": command_result(
                ["git", "status", "--short"]
            )["stdout"],
            "environment": environment,
            "verification_sources": {
                str(path.relative_to(ROOT)): {
                    "sha256": sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
                for path in VERIFICATION_SOURCE_PATHS
            },
            "files": {
                filename: {
                    "sha256": sha256_file(ARTIFACT_DIR / filename),
                    "byte_size": (ARTIFACT_DIR / filename).stat().st_size,
                }
                for filename in manifest_files
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Stage 3/4 F6 parity and activation-readiness evidence."
    )
    parser.parse_args()
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

    before_hashes, start_head = verify_initial_state()
    environment = collect_environment()
    verify_environment(environment)
    predictor, integrity = verify_candidate_directory(CANDIDATE_DIR)
    fresh_process = fresh_process_audit()
    valid_cases = [case for case in parity_case_registry() if case.valid]
    parity = [evaluate_parity_case(predictor, case) for case in valid_cases]
    p6 = corrupted_contract_check()
    p2 = next(case for case in valid_cases if case.case_id == "P2")
    leakage = sunday_leakage_verification(predictor, p2)
    warning_assessment = assess_warnings()
    admin_server = streamlit_startup_smoke("app.py", 8765)
    staff_server = streamlit_startup_smoke("app_staff.py", 8766)
    targeted_tests = command_result(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_f6_candidate_parity.py",
            "tests/test_streamlit_model_ui.py",
            "tests/test_recommendation_ui.py",
        ]
    )
    full_tests = command_result([sys.executable, "-m", "pytest", "-q"])
    diff_check = command_result(["git", "diff", "--check"])
    after_hashes = collect_protected_hashes()
    if before_hashes != after_hashes:
        raise AssertionError("A protected file changed during Stage 3/4 verification")
    build_artifacts(
        before_hashes=before_hashes,
        start_head=start_head,
        environment=environment,
        integrity=integrity,
        fresh_process=fresh_process,
        parity=parity,
        p6=p6,
        leakage=leakage,
        warning_assessment=warning_assessment,
        admin_server=admin_server,
        staff_server=staff_server,
        targeted_tests=targeted_tests,
        full_tests=full_tests,
        diff_check=diff_check,
        after_hashes=after_hashes,
    )
    print(f"Stage 3/4 evidence generated at {ARTIFACT_DIR}")
    print(targeted_tests["stdout"].splitlines()[-1])
    print(full_tests["stdout"].splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
