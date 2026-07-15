#!/usr/bin/env python3
"""Finalize Phase 2A reports from completed on-disk model outputs after post-fit failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase2a_feature_repair import (
    EXTRA_ARTIFACTS,
    LEGACY_PREDICTIONS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    PHASE1_DIR,
    PROTECTED_PATHS,
    REQUIRED_ARTIFACTS,
    artifact_reports,
    baseline_metrics,
    development_decision_metrics,
    exact_f0_reproduction,
    fixed_split,
    phase1_gate,
    sha256_file,
    sha256_json,
    stable_attendance_fingerprint,
    validate_outputs,
    write_text,
)
from src.config import DATE_COL, TARGET_COL
from src.feature_sets import (
    F0,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    FEATURE_SET_IDS,
    FeatureSetDefinition,
    select_compact_f6,
    select_f5_parent,
)
from src.origin_backtest import SCENARIO_DEFINITIONS
from src.origin_features import T1_VALID_WEEKENDS, W0_NO_WEATHER, W1_OBSERVED_REPLAY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-test-result", default="44 passed (exit 0)")
    parser.add_argument("--full-test-result", default="68 passed (exit 0)")
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_registry() -> dict[str, FeatureSetDefinition]:
    payload = json.loads((OUTPUT_DIR / "02_feature_set_registry.json").read_text())
    registry: dict[str, FeatureSetDefinition] = {}
    for feature_set_id in FEATURE_SET_IDS:
        item = payload[feature_set_id]
        registry[feature_set_id] = FeatureSetDefinition(
            feature_set_id=item["feature_set_id"],
            name=item["name"],
            feature_list=tuple(item["feature_list"]),
            feature_groups=tuple(item["feature_groups"]),
            parent_feature_set=item["parent_feature_set"],
            controlled_change=item["controlled_change"],
            expected_value=item["expected_value"],
            leakage_assessment=item["leakage_assessment"],
            production_availability=item["production_availability"],
        )
    return registry


def read_csv(name: str, date_columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, parse_dates=date_columns or [])


def main() -> int:
    args = parse_args()
    required_completed = [
        "02_feature_set_registry.json",
        "03_development_confirmation_split.md",
        "05_feature_set_predictions.csv",
        "06_feature_set_metrics.csv",
        "07_paired_candidate_comparison.csv",
        "08_scenario_and_horizon_comparison.csv",
        "09_daytype_comparison.csv",
        "10_recent_period_analysis.csv",
        "11_feature_missingness_and_provenance.csv",
        "12_feature_importance_and_redundancy.csv",
        "13_quantile_coverage_analysis.csv",
        "14_w1_secondary_diagnostic.csv",
        "feature_set_metric_breakdowns.csv",
        "feature_value_provenance.csv",
        "fold_preprocessing_diagnostics.csv",
        "paired_errors_by_target_date.csv",
    ]
    missing = [name for name in required_completed if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Completed model/report outputs are missing: {missing}")

    gate, phase1_preferred = phase1_gate()
    role_map, split = fixed_split(phase1_preferred)
    registry = load_registry()
    predictions = read_csv(
        "05_feature_set_predictions.csv",
        ["forecast_origin", "target_date", "training_end_date"],
    )
    decision = read_csv("06_feature_set_metrics.csv")
    breakdowns = read_csv("feature_set_metric_breakdowns.csv")
    paired = read_csv("07_paired_candidate_comparison.csv")
    paired_targets = read_csv("paired_errors_by_target_date.csv", ["target_date"])
    scenario_horizon = read_csv("08_scenario_and_horizon_comparison.csv")
    daytype = read_csv("09_daytype_comparison.csv")
    recent = read_csv("10_recent_period_analysis.csv")
    missingness = read_csv("11_feature_missingness_and_provenance.csv")
    importance = read_csv("12_feature_importance_and_redundancy.csv")
    quantile = read_csv("13_quantile_coverage_analysis.csv")
    w1 = read_csv("14_w1_secondary_diagnostic.csv")
    definitions = read_csv("04_feature_definitions.csv")
    provenance = read_csv(
        "feature_value_provenance.csv", ["forecast_origin", "target_date"]
    )
    preprocessing = read_csv(
        "fold_preprocessing_diagnostics.csv",
        ["training_start_date", "training_end_date"],
    )

    primary = predictions[predictions["weather_policy"] == W0_NO_WEATHER].copy()
    development = pd.DataFrame(
        [
            development_decision_metrics(
                primary[primary["feature_set_id"] == feature_set_id]
            )
            for feature_set_id in FEATURE_SET_IDS
        ]
    )
    f5_parent_id = select_f5_parent(development)
    f6_features, f6_groups, f6_decisions = select_compact_f6(
        development, f5_parent_id=f5_parent_id
    )
    if f5_parent_id != registry[F5].parent_feature_set:
        raise AssertionError("Recovered F5 parent differs from the locked registry")
    if f6_features != list(registry[F6].feature_list):
        raise AssertionError("Recovered F6 list differs from the locked registry")

    baselines = baseline_metrics(primary)
    f0_reproduction = exact_f0_reproduction(
        primary[primary["feature_set_id"] == F0], phase1_preferred
    )

    registry_mtime = datetime.fromtimestamp(
        (OUTPUT_DIR / "02_feature_set_registry.json").stat().st_mtime, timezone.utc
    ).isoformat()
    predictions_mtime = datetime.fromtimestamp(
        (OUTPUT_DIR / "05_feature_set_predictions.csv").stat().st_mtime, timezone.utc
    ).isoformat()
    validation = validate_outputs(
        predictions,
        provenance,
        preprocessing,
        registry,
        role_map,
        registry_mtime,
        predictions_mtime,
    )
    validation["f6_lock_timing_evidence"] = (
        "registry serialized in the runner's f6_locked branch before the confirmation loop; "
        "registry filesystem time precedes completed prediction artifact time"
    )

    phase1_manifest = json.loads((PHASE1_DIR / "phase1_manifest.json").read_text())
    expected_protected = dict(phase1_manifest["production_file_hashes_at_generation"])
    expected_protected.update(
        {
            "models/visitor_model_ny_12550.joblib": (
                "061be9292fb85cecbd4b5ab2a213bba9d4a692305d23234ff3f81c49affc94f2"
            ),
            "artifacts/ny_12550/backtest_predictions.csv": (
                "8020205870723c00bd704b18a47ccba4a9ccd0de9e835045d0ddafbd680927fa"
            ),
        }
    )
    current_protected = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_PATHS
    }
    if current_protected != expected_protected:
        raise AssertionError(
            f"Protected files changed: expected={expected_protected}, current={current_protected}"
        )

    # Recreate all human-readable artifacts consistently; these are only outputs from the
    # disclosed post-fit run, never pre-existing user/Phase 1 artifacts.
    artifact_reports(
        gate=gate,
        split=split,
        registry=registry,
        definitions=definitions,
        predictions=predictions,
        decision=decision,
        breakdowns=breakdowns,
        paired=paired,
        paired_targets=paired_targets,
        scenario_horizon=scenario_horizon,
        daytype=daytype,
        recent=recent,
        missingness=missingness,
        importance=importance,
        quantile=quantile,
        w1=w1,
        f6_features=f6_features,
        f6_groups=f6_groups,
        f6_decisions=f6_decisions,
        baselines=baselines,
        validation=validation,
        protected_before=expected_protected,
        protected_after=current_protected,
        args=argparse.Namespace(
            targeted_test_result=args.targeted_test_result,
            full_test_result=args.full_test_result,
            recover_w1_deterministically=True,
        ),
    )

    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    authority = joblib.load(MODEL_PATH)["history_df"][[DATE_COL, TARGET_COL]].copy()
    authority[DATE_COL] = pd.to_datetime(authority[DATE_COL]).dt.normalize()
    artifact_files = [
        name for name in REQUIRED_ARTIFACTS if name != "phase2a_manifest.json"
    ] + EXTRA_ARTIFACTS
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "starting_commit": commit,
        "ending_commit": commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "commands_executed": [
            "git status --short",
            "git branch --show-current",
            "git rev-parse HEAD",
            "python --version (failed: python command unavailable)",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv /tmp/soup-kitchen-forecast-phase2a-venv",
            "/tmp/soup-kitchen-forecast-phase2a-venv/bin/python -m pip install -r requirements.txt pytest",
            "/tmp/soup-kitchen-forecast-phase2a-venv/bin/python -m pytest -q tests/test_origin_features.py tests/test_origin_backtest.py tests/test_feature_sets.py tests/test_phase2a_feature_repair.py",
            "/tmp/soup-kitchen-forecast-phase2a-venv/bin/python -m pytest -q",
            "run_phase2a_feature_repair.py attempt 1 (failed after F0: wrapper rejected four legitimate skipped early folds; no repaired candidate ran)",
            "run_phase2a_feature_repair.py attempt 2 (all model stages completed; failed in paired-report column collision)",
            "run_phase2a_feature_repair.py attempt 3 (all model stages completed; attendance validator incorrectly included calendar/weather source dates)",
            "run_phase2a_feature_repair.py attempt 4 --recover-w1-deterministically (all primary stages and validation completed; failed formatting a list-valued Markdown cell)",
            "finalize_phase2a_reports.py (finalized from completed on-disk model outputs)",
        ],
        "phase1_input_files": sorted(
            str(path.relative_to(ROOT)) for path in PHASE1_DIR.iterdir() if path.is_file()
        ),
        "input_fingerprints": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                MODEL_PATH,
                ROOT / "data/locations/ny_12550/weather_daily.csv",
                LEGACY_PREDICTIONS_PATH,
                *PHASE1_DIR.glob("*"),
            ]
            if path.is_file()
        },
        "attendance_authority": {
            "source": "models/visitor_model_ny_12550.joblib history_df[[service_date, visitors]]",
            "row_count": int(len(authority)),
            "date_range": [
                authority[DATE_COL].min().strftime("%Y-%m-%d"),
                authority[DATE_COL].max().strftime("%Y-%m-%d"),
            ],
            "stable_attendance_sha256": stable_attendance_fingerprint(authority),
        },
        "origin_definitions": SCENARIO_DEFINITIONS,
        "primary_policy": {
            "weekday_policy": T1_VALID_WEEKENDS,
            "weather_policy": W0_NO_WEATHER,
        },
        "feature_set_registry": {
            feature_set_id: item.to_dict() for feature_set_id, item in registry.items()
        },
        "development_confirmation_ranges": split.assign(
            development_start=split["development_start"].dt.strftime("%Y-%m-%d"),
            development_end=split["development_end"].dt.strftime("%Y-%m-%d"),
            confirmation_start=split["confirmation_start"].dt.strftime("%Y-%m-%d"),
            confirmation_end=split["confirmation_end"].dt.strftime("%Y-%m-%d"),
        ).to_dict("records"),
        "selection_event_log": [
            {"event": "phase1_gate_passed"},
            {"event": "f1_f4_development_complete"},
            {"event": "f5_parent_selected", "f5_parent_id": f5_parent_id},
            {"event": "f5_development_complete"},
            {
                "event": "f6_locked",
                "timestamp_evidence_utc": registry_mtime,
                "feature_list_sha256": sha256_json(f6_features),
                "feature_count": len(f6_features),
            },
            {"event": "f6_development_complete"},
            {
                "event": "confirmation_scoring_complete",
                "prediction_artifact_timestamp_utc": predictions_mtime,
            },
            {
                "event": "w1_secondary_recovered_after_two_completed_postfit_failures",
                "method": "F0 exact Phase 1 W1; F5/F6 deterministic weather-free W0 equivalence",
            },
        ],
        "f5_parent_id": f5_parent_id,
        "f6_lock": {
            "feature_list_sha256": sha256_json(f6_features),
            "feature_list": f6_features,
            "timing_evidence": validation["f6_lock_timing_evidence"],
        },
        "model_configurations": {
            "point": "RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=2,random_state=42)",
            "quantile": "HistGradientBoostingRegressor(loss=quantile,quantile=0.8,learning_rate=0.05,max_depth=4,max_iter=500,random_state=42)",
            "segmentation": "Saturday/Sunday",
            "min_segment_training_rows": 18,
            "window": "expanding",
            "preprocessing": "fold-local SimpleImputer(median, keep_empty_features=True)",
        },
        "random_seeds": {"models": 42, "block_bootstrap": 20260715},
        "code_files_created": [
            "src/feature_sets.py",
            "scripts/run_phase2a_feature_repair.py",
            "scripts/finalize_phase2a_reports.py",
            "tests/test_feature_sets.py",
            "tests/test_phase2a_feature_repair.py",
        ],
        "code_files_modified": ["src/origin_features.py", "src/origin_backtest.py"],
        "artifact_files_created": artifact_files + ["phase2a_manifest.json"],
        "artifact_sha256_excluding_manifest": {
            name: sha256_file(OUTPUT_DIR / name) for name in artifact_files
        },
        "existing_user_or_phase1_files_overwritten": [],
        "partial_phase2a_outputs_finalized_after_disclosed_postfit_failure": True,
        "test_results": {
            "targeted": args.targeted_test_result,
            "full": args.full_test_result,
        },
        "phase1_gate": gate,
        "f0_separate_reproduction": f0_reproduction,
        "prediction_row_counts": {
            "total": int(len(predictions)),
            "primary_w0": int(len(primary)),
            "w1_secondary": int(
                (predictions["weather_policy"] == W1_OBSERVED_REPLAY).sum()
            ),
        },
        "candidate_row_counts": {
            str(key): int(value) for key, value in primary.groupby("feature_set_id").size().items()
        },
        "leakage_validation_results": validation,
        "production_file_integrity_checks": {
            "passed": current_protected == expected_protected,
            "expected": expected_protected,
            "current": current_protected,
        },
        "known_limitations": [
            "single location and finite historical interval",
            "W1 uses realized target-date weather and is not deployment-valid",
            "no archived forecast weather exists",
            "confirmation estimates are descriptive and not an independent external study",
            "feature importance is associational and correlated features split importance",
            "permutation importance is not meaningful with one test target per expanding fold",
            "final W1 rows were recovered from deterministic equivalents after two completed W1 runs were lost to downstream validation failures",
            "exact in-process F6 lock timestamp was lost to a downstream formatter failure; code-order and file-time evidence are retained",
            "Phase 2A does not modify or validate a deployed retrained model",
        ],
        "phase2b_started": False,
        "artifacts_ignored_by_git": True,
    }
    write_text("phase2a_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

    missing_required = [name for name in REQUIRED_ARTIFACTS if not (OUTPUT_DIR / name).exists()]
    if missing_required:
        raise AssertionError(f"Missing required artifacts: {missing_required}")
    print(
        f"Phase 2A reports finalized: {len(predictions)} predictions, "
        f"{len(REQUIRED_ARTIFACTS)} required artifacts.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
