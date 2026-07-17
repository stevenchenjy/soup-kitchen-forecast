# F6 Stage 3/4 Parity and Activation Readiness

## Decision

Stage 3/4 verification passed for the exact
`ny_12550_f6_2026-07-12_v1` candidate. The candidate is technically ready for
a separately authorized, controlled activation only after the deployment
runtime is pinned and nightly retraining is made F6-safe. It is not ready for
immediate activation in the current deployment configuration.

No activation, deployment, retraining, Supabase write, production-attendance
change, nightly-workflow change, or model-package rewrite occurred in this
stage.

## Candidate and protected state

- Candidate package SHA-256:
  `9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397`
- Candidate package ID: `ny_12550_f6_2026-07-12_v1`
- Package schema: v2
- Feature set: `F6_COMPACT_SELECTED`, 33 features
- Feature-order SHA-256:
  `dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419`
- Recommendation policy: `C0_EXISTING_RAW_QUANTILE`
- Candidate status: `candidate_not_active`
- Active-model SHA-256 before and after:
  `ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0`
- Nightly-workflow SHA-256 before and after:
  `fb6df61cbccbaea809146c87f24a81f342642f1fc695fcace8f1fa50a0f19225`

The candidate checksum manifest, semantic metadata, tracked contract,
Saturday/Sunday point models, quantile models, preprocessors, and embedded
history through `2026-07-12` all passed validation. A fresh subprocess loaded
and predicted without consulting the local source CSV or ignored research
artifacts.

## Deterministic parity results

All raw feature, transformed feature, point-prediction, and quantile-prediction
differences were exactly `0.0`; the acceptance tolerance remained `1e-10`.

| Case | Target | Origin | Horizon | Segment | Point | Q80 | C0 meals |
|---|---|---|---:|---|---:|---:|---:|
| P1 | 2026-07-18 | 2026-07-17 | 1 | Saturday | 100.96578389992362 | 115.54079802971415 | 116 |
| P2 | 2026-07-19 | 2026-07-17 | 2 | Sunday | 168.25556845251214 | 168.4604402409751 | 169 |
| P3 | 2026-07-26 | 2026-07-20 | 2 | Sunday | 173.53999714535956 | 177.84103442987765 | 178 |
| P4 | 2026-08-01 | 2026-07-17 | 5 | Saturday | 101.81444344177068 | 113.15611751231293 | 114 |
| P5 | 2023-01-29 | 2023-01-27 | 2 | Sunday | 136.44032350750385 | 148.71919132121644 | 149 |

Every case used the correct segment preprocessor, produced 33 finite
transformed values, remained nonnegative and within the UI's operational
attendance domain, returned `ceil(Q80)`, and kept both percentage and residual
buffers at `0.0`. P6, a
corrupted feature contract, failed before prediction.

For the Sunday-before-weekend case, all attendance provenance was Sunday-only
and no source date exceeded the Friday origin. Appending a post-origin Saturday
sentinel value of 9999 and then masking it changed no feature or prediction.
The Sunday service horizon remained 2.

## Website verification

Streamlit AppTest exercised `app.py` and `app_staff.py` with deterministic test
users and mutation tripwires:

- F6 Saturday and Sunday recommendations matched `VisitorPredictor` exactly.
- Package ID, schema v2, and `F6_COMPACT_SELECTED` were rendered.
- The percentage-buffer slider was absent and raw-Q80 wording was rendered.
- The F6 weather client was never constructed.
- No source attendance CSV, Supabase write, production prediction-log write,
  or rendered secret was observed.
- The active schema-v1 model still loaded, retained its percentage-buffer
  slider and legacy calculation, and was not mislabeled as F6.

A manual in-app browser session was not available in this environment. AppTest
executed the real Streamlit scripts; a supplemental server socket smoke was
skipped because the verification sandbox denied local socket binding.

## Warning assessment

The candidate was loaded repeatedly under both Python 3.13.2 and an isolated
Python 3.12.6 environment with NumPy 2.5.1, pandas 3.0.3,
scikit-learn 1.5.2, and joblib 1.4.2. Each load emitted 5,482 identical
joblib/NumPy deprecation warnings while restoring array shapes. Transformation
and prediction emitted zero warnings, and all repeated signatures were
identical across runtimes.

Classification: **deployment concern requiring environment pin**. This is not
evidence of candidate corruption, but a future NumPy release could remove the
deprecated behavior. Pin `numpy==2.5.1` before activation and record the other
verified package versions. Do not reserialize the candidate merely to suppress
the warning.

## Activation prerequisites and rollback

Two hard gates remain before a future activation:

1. Pin the deployed NumPy/runtime versions to the verified environment.
2. Make nightly retraining F6-aware or explicitly exclude `ny_12550`. The
   current nightly path trains schema v1 and would overwrite an activated F6
   package on the next dirty run.

The ignored Stage 3/4 evidence bundle contains a complete, unexecuted runbook
for external backup, same-directory temporary validation, atomic
`os.replace()`, active-path hash verification, Saturday/Sunday/H5 and website
smokes, production-log checks, and exact hash-verified rollback. The backup
must remain outside the repository because a repository backup path is not
currently ignored.

## Verification record

- Focused Stage 3/4 tests: 46 passed
- Complete suite: 159 passed, 30 skipped, 14 warnings, 9 subtests passed
- `git diff --check`: passed
- Required ignored evidence bundle:
  `artifacts/ny_12550/production_upgrade/stage3_4_parity_and_web/`

The exact active and candidate binaries remained unchanged, and the candidate
remains inactive.
