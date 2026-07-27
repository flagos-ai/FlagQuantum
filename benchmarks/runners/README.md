# Benchmark runners

This namespace is the migration target for executable benchmark drivers. A
runner must be deterministic from its CLI arguments, emit a self-contained
JSON payload, and avoid importing plotting or exploratory research modules.

See [`MIGRATION_LOG.md`](MIGRATION_LOG.md) for adapter-by-adapter status.

Library consumers can use the stable package API:

```python
import benchmarks.runners as runners

print(runners.names())
runner_main = runners.resolve("environment_probe")
```

Existing scripts in `benchmarks/` remain compatibility entry points during the
migration. New runners should be added here first; old scripts are moved only
after their import and output contracts have tests.

Smoke example from the repository root:

```bash
python -m benchmarks.runners.environment_probe \
  --json-output benchmarks/results/environment_probe.json
```

The same runner can be invoked as `python benchmarks/runners/environment_probe.py`.

The discoverable package entry point is:

```bash
python -m benchmarks.runners environment_probe \
  --json-output benchmarks/results/environment_probe.json
```

Migration checklist for an existing top-level script:

1. preserve its historical CLI as a compatibility wrapper;
2. emit `runner` and versioned `schema` through `contract.py`;
3. use `write_json_atomic` for result files;
4. register the new entry in `registry.py`;
5. add a smoke and output-contract test before removing duplicate logic.

Runner names must use lowercase `snake_case` (for example,
`environment_probe`); duplicate registrations are rejected.
Every runner payload must include a versioned schema name ending in `.vN`,
such as `flagquantum.benchmark.environment.v1`.

`statevector_weak_scaling_report.py` is the first legacy driver with a
package-safe import boundary. It remains at its historical path while the
contract-aware adapter below serves as the new entry point.

Its contract-aware adapter is now available as:

```bash
python -m benchmarks.runners statevector_weak_scaling \
  benchmarks/results/weak_1.json benchmarks/results/weak_2.json \
  --json-output benchmarks/results/weak_scaling_report.v1.json
```

The matching strong-scaling adapter uses the same contract:

```bash
python -m benchmarks.runners statevector_strong_scaling \
  benchmarks/results/strong_1.json benchmarks/results/strong_2.json \
  --json-output benchmarks/results/strong_scaling_report.v1.json
```

Training-scaling reports use the same adapter pattern:

```bash
python -m benchmarks.runners statevector_training_scaling \
  benchmarks/results/training_1.json benchmarks/results/training_2.json \
  --json-output benchmarks/results/training_scaling_report.v1.json
```
