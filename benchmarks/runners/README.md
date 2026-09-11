# Maintained benchmark runners

This package backs the installed `flagquantum-benchmark` command. It owns
scenario discovery, argument forwarding, environment capture, atomic JSON
output, and versioned result contracts.

```bash
flagquantum-benchmark list
flagquantum-benchmark info environment_probe
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json
```

Library tooling may inspect the lazy registry without importing heavy
benchmark implementations:

```python
import flagquantum.benchmarking as runners

for spec in runners.specs():
    print(spec.name, spec.hardware)
```

Runner names use lowercase `snake_case`. Each maintained runner must:

1. be deterministic from its explicit arguments;
2. emit a self-contained, versioned JSON payload;
3. write results atomically through `contract.py`;
4. avoid importing plotting or exploratory research modules;
5. include CLI and output-contract tests.

Top-level modules under `benchmarks/` are implementation details during the
registry migration. Research and historical evidence code must stay outside
this supported namespace.
