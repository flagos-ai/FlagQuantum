# QEC memory declarations

Core owns code-driven source strings, measurement handles, detector and observable
layouts, and `MemoryCircuit`. These declarations may use standard-library helpers,
QEC code records, Pauli algebra and logical-product certification. They must not
lower programs, execute shots, sample noise, construct DEMs or import execution,
compiler, runtime or simulation infrastructure.

`memory.py` contains the original declaration implementation. Existing imports
from `flagquantum.qec` and `flagquantum.qec.circuit` resolve to exactly the same
records and builder as this package. Class, function and method metadata retain
`flagquantum.qec.circuit` for pickle and caller-owned safe-global identifiers.
Source inspection must follow `memory.py` or a method's code filename, rather
than assuming that historical `__module__` identifies the physical source file.
The QEC root facade is still eager; this boundary makes no cold-root-import claim.

Core owns the compatibility aliases. No removal release is approved; their
removal requires a separate approved public-API and serialized-artifact migration.

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
an unchanged historical pickle class, zero noiseless detection events and zero
logical error in each of the three existing feedback modes. The core-only
feedback example remains `examples/qec/restructure_compatibility.py`.
