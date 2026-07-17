# F6 Stage 5 Production Activation

On 2026-07-16, the exact verified package
`ny_12550_f6_2026-07-12_v1` was atomically activated for `ny_12550`.

- Active path: `models/visitor_model_ny_12550.joblib`
- Active SHA-256: `9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397`
- Schema: 2
- Feature set: `F6_COMPACT_SELECTED` (33 features)
- Feature-order SHA-256:
  `dac868ae1a739cbee55443a953c6ab5c45876e158e40b57300ffe1c9607f7419`
- Recommendation policy: `C0_EXISTING_RAW_QUANTILE`
- Versioned legacy backup:
  `models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib`
- Backup SHA-256:
  `ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0`

The candidate itself was not retrained, modified, or reserialized. Activation
used `scripts/activate_f6_candidate.py`, a validated sibling temporary file,
and `os.replace`. The fallback `models/visitor_model.joblib` remained unchanged.

Nightly retraining is F6-aware for `ny_12550`: it trains from the existing
attendance loader into a temporary schema-v2 package, validates the locked
contract in a fresh process, retains the previous active F6 package, and only
then performs an atomic replacement. Other locations retain the legacy path.

## Rollback

Run from the repository root with the pinned virtual environment:

```bash
.venv/bin/python scripts/activate_f6_candidate.py \
  --mode rollback \
  --candidate models/candidates/ny_12550_f6_2026-07-12_v1/model_package.joblib \
  --active models/visitor_model_ny_12550.joblib \
  --backup models/backups/ny_12550_schema1_pre_f6_2026-07-16.joblib \
  --expected-candidate-sha256 9eb8c75271c301f3f44ac864705c23a779c0a9f3fadedcfe896d5dea350e3397 \
  --expected-legacy-sha256 ee56a3fb03c212653a97f6073600189a51592db355efabe09ef2b138f36976f0 \
  --package-id ny_12550_f6_2026-07-12_v1 \
  --receipt artifacts/ny_12550/production_upgrade/stage5_activation/rollback_receipt.json
```

The rollback command refuses hash, schema, or package-ID mismatches and uses an
atomic replacement. A temporary-copy rollback rehearsal restored the exact
legacy package, performed a legacy smoke prediction, and then reapplied the
exact F6 candidate with matching reference predictions.
