# Runner migration log

| Adapter | Historical implementation | Status | Contract test |
| --- | --- | --- | --- |
| `environment_probe` | new | active | `test_benchmark_environment_probe.py` |
| `statevector_strong_scaling` | `statevector_scaling_report.py` | dual-entry | `test_statevector_strong_scaling_runner.py` |
| `statevector_weak_scaling` | `statevector_weak_scaling_report.py` | dual-entry | `test_statevector_weak_scaling_runner.py` |
| `statevector_training_scaling` | `statevector_training_scaling_report.py` | adapter registered | `test_scaling_runner_script_entrypoints.py` |

“Dual-entry” means the historical script remains untouched while the adapter
owns the versioned runner envelope. Removal requires one release with import,
payload, and reproducibility checks passing.

Runtime namespace migrations are enforced by:

```bash
python tools/check_architecture.py
```

The checker rejects both reintroduction of the removed compatibility package
and imports from its former namespace.
