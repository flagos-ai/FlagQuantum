# QEC memory declarations

Core owns code-driven source strings, measurement handles, detector and observable
layouts, and `MemoryCircuit`. These declarations may use standard-library helpers,
QEC code records, Pauli algebra and logical-product certification. They must not
lower programs, execute shots, sample noise, construct DEMs or import execution,
compiler, runtime or simulation infrastructure.

`memory.py` contains the declaration implementation. Root and canonical
`flagquantum.qec.circuits.memory` imports share records and builders. Their
class/function/method pickle metadata now uses the canonical module. The approved
0.4.0 retirement removes `qec.circuit` and its old pickle lookups; caller-owned
safe-global lists must use the canonical objects. No custom reducer is added.
The QEC root remains eager; this boundary makes no cold-root-import claim.

For a layout change, edit `memory.py`, retain check/measurement order and exact
source anchors, and run:

```bash
python -m pytest tests/qec/test_memory_circuit.py tests/qec/test_memory_circuit_execution.py tests/qec/test_restructure_execution.py -q
```

For a short user workflow covering declarations, maps, real sampling and feedback:

```bash
pip install -e '.[stim]'
PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py
```

It reports `9 8 1` measurements/detectors/observables, an `(8, 9)` detector map,
the canonical declaration pickle class, zero noiseless detection events and zero
logical error in each of the three existing feedback modes. The core-only
feedback example remains `examples/qec/restructure_compatibility.py`.
