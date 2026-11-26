# Pending #1084 import reconciliation

Only `model.py` and `verifier.py` are retained for the audited #1084 tests
`test_construction_register_width_model.py` and `test_drawer_register_score.py`.
They forward to `compiler/frontends/_structured/`; this namespace exposes no
frontend workflow. Integration owns removal when those incoming imports are
updated during #1084 integration. This is an internal merge dependency, not a
public compatibility promise.
