# Signed quantum sawtooth map

Open `walkthrough.ipynb` for the complete native FlagQuantum workflow.

```sh
python -m examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.overview.main
```

```python
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.overview.workflow import run_workflow

for row in run_workflow(shots=1000, seed=1970):
    print(row.evolution.kicks, row.observations)
```

Retain three signed bits, initial momentum -2, literal kick strength 0.1 and
all four original times 2, 4, 6 and 8. Each receives 1,000 actual local draws.
All sampled P(-2) values exceed 0.85. Independent Circuit full states check
native evolution and exact CX/U3 conversion, including absolute phase.
The source one-step resource extrapolation is labeled as an estimate and
reported separately from actual converted full-circuit resources. Counts refer
to the exact CX/U3 conversion stage, before Runtime compilation and its
additional gate optimizations.

The original hardware switch is false. The included CSV files are historical
upstream IonQ observations, verified by hash and labeled accordingly. They do
not represent newly submitted FlagQuantum hardware jobs. The source plot's +4
point is zero padding outside the representable signed range -4..3.

See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json) for scope,
source mapping and asset hashes. [Third-party license](../../../../../CLASSIQ_LICENSE.txt).
