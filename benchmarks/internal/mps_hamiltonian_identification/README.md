# MPS Hamiltonian identification

Recover spatially varying couplings and fields from time-resolved observations
using a differentiable MPS model. The scientific objective is parameter
identification and held-out prediction, not merely fitting training data.

## Start on CPU

From an installed checkout:

```bash
python benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 8 --n-initial-states 2 --time-steps 1,2 \
  --observation-stride 2 --max-bond 8 --steps 3 \
  --device cpu --output /tmp/fq-mps-system-id-smoke.json
```

Check recovered parameters and held-out observables across observation layouts,
time depths, and truncation settings. Low observation error alone does not
establish a uniquely identified Hamiltonian.

[Experiment guide and measured results](EXPERIMENT.md) preserves GPU commands,
compiled and site-sharded paths, historical measurements, and their scope.
Probe parallelism and site-sharded capacity are different execution semantics;
use the [benchmark evidence policy](../../results/README.md) when making claims.
