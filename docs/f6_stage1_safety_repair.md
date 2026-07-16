# F6 Stage 1.1 Safety Repair

## Baseline

- Starting `origin/main`: `b162c6e28dd65182e7eda74d5d41ac5fc15930c5`
- Active `ny_12550` model SHA-256: `ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0`
- Legacy fallback model SHA-256: `cca9b22d63d85ff0a4f0ebd14e09209d1dfffa73f0f63e93d9117d93b75bd920`
- Nightly workflow SHA-256: `fb6df61cbccbaea809146c87f24a81f342642f1fc695fcace8f1fa50a0f19225`
- Predictor SHA-256: `1cbfc95609dc13f525f99f4d78ce7d07cf497d4bdd3971ad1b9667e9eed39c29`
- Active training entrypoint SHA-256: `70b07ef103c58137c97ca25143e23cb8dddebb1d2680e9cfb0713d572458ec09`
- Local model-optimization artifact tree fingerprint: `c0122c81a22a96a991b40089ab7932c1319ca4ecb6e03612686f21672ac94fc9`
- Local `data/locations/ny_12550/Updated` directory: absent
- Tracked local attendance database SHA-256: `d4b0df65bebac69fe3069199cc71d062c2eea956102aafaf66425c1ce8a30d9d`
- Location registry SHA-256: `f2aa6b2a2aa71ed20b1da53432bf8f2f338093a291445438c3e9f727a036f67d`

## Confirmed issues

1. `src/production_features.py` requires an ignored Phase 2A.5 research lock file. A clean checkout therefore cannot validate the F6 production contract.
2. `tests/test_production_f6_integration.py` requires an ignored production-derived Supabase CSV.
3. `scripts/train_backtest.py`, which is called by nightly retraining, currently trains schema-v2 F6 models directly into the active schema-v1 model path.
4. The active `ny_12550` model is a schema-v1 package and must remain unchanged until the later activation stage.

## Intended repair

1. Add a single tracked, deterministically validated F6 contract at `config/model_contracts/f6_v1.json`. Production training and prediction will read this resource. Research lock files will be optional diagnostics only.
2. Replace local-artifact and production-export test dependencies with tracked contract data, synthetic attendance fixtures, and temporary model packages. Optional research-integrity checks will skip when their source is absent.
3. Restore `scripts/train_backtest.py` as the legacy active/nightly training entrypoint. Add a separate `scripts/train_f6_candidate.py` command that requires explicit input and output arguments, writes a versioned candidate directory, records metadata and checksums, refuses active paths, and refuses overwrite.
4. Preserve schema-v1 and schema-v2 predictor support. Schema-v2 packages will be checked against the tracked contract and locked C0 recommendation policy.
5. Keep the percentage-buffer control for legacy packages. Hide it for F6/C0 packages, label the raw 80th-percentile recommendation, and show package/schema identity.
6. Add a credential-free clean-checkout test workflow that is separate from nightly retraining.

No F6 model will be trained, activated, or deployed as part of this repair.
