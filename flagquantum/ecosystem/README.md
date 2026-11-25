# Ecosystem interoperability

`flagquantum.ecosystem` is the interoperability boundary. It translates objects
from optional external frameworks into FlagQuantum-owned `CircuitIR` and
translates owned programs back only when explicitly requested. Conversion
reports make every unsupported or lossy semantic difference visible.

Interop does not define a second canonical IR, compile programs, implement
numerical kernels, or let framework objects enter Core, Compiler, Runtime, or
Simulation. An explicitly selected local external simulator may execute an
owned program inside its framework package and must return a FlagQuantum-owned
result. Importing this package and listing its adapters must not import an
optional framework.

## Where to start

- `contracts.py`: framework-neutral adapter and conversion-report protocol.
- `registry.py`: immutable, lazy adapter discovery.
- `conformance.py`: common round-trip and rejection checks.
- `_semantic_parity.py`: deterministic circuits shared by optional SDK parity lanes.
- `<framework>/conversion.py`: external object to/from `CircuitIR` conversion.
- `<framework>/models.py`: adapter-local results and errors.
- `<framework>/adapter.py`: the small implementation of the common protocol.
- `<framework>.run`: the consistent explicit entry point when that framework
  has a qualified local execution bridge.

Framework-specific local execution entry points use
`flagquantum.ecosystem.<framework>.run` and reuse native `fq.run` vocabulary:
`outputs`, `ExecutionOptions`, `samples`, and `counts`. Remote services and
provider hardware remain owned by Remote; an Ecosystem runner must not become
an implicit router or fallback path.

## Ten-minute change path

For a small adapter conversion change, modify one framework adapter, add its
round-trip or explicit-rejection case, and run:

```bash
python -m pytest tests/team/ecosystem tests/api_contract/test_open_source_golden_paths.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Framework-specific dependencies remain optional and isolated under their
adapter package. Changes to the candidate-stable interop protocol, signatures,
issue schema, or registry semantics require the public API change process;
ordinary framework mappings should not extend that shared protocol.


## Native construction source (private)

`_qmod/` adapts a bounded, pinned native Qmod subset into verified Compiler
construction records before gate-level IR exists. This private handoff differs
from the public CoreIR adapter protocol; it does not extend that protocol or
register another framework. Compiler retains lowering/selection and Runtime
retains execution. See its [contributor guide](_qmod/README.md) and
[separate compatibility counts](../../docs/roadmap/CS07_QMOD_PARSER_DELIVERY.md).
No public native-source execution entry is admitted by this private work.

## Optional OpenFermion operator migration

`openfermion.py` owns only explicit operator conversion. Its operator entry points
are `from_openfermion_fermion`, `to_openfermion_fermion`,
`from_openfermion_qubit`, `to_openfermion_qubit`, and the one-way
`from_openfermion_interaction` import. Imports stop at native
fermionic/Pauli algebra and standard-library validation; upstream OpenFermion and
its NumPy tensor boundary are loaded only when explicitly needed by a conversion.
It must not own simulation, model construction, encoding, chemistry drivers or
runtime selection. Native modules
never depend on this adapter. No adapter registry entry or root API is added.

For an old-object migration, install `openfermion==1.7.1` separately in the
interoperability environment; upstream also installs its own dependencies,
including Cirq. Neither package is required by the native core. This experimental
boundary is verified against that upstream release; it does not promise complete
upstream API compatibility. See the
[EXT05 contract](../../docs/development/OPENFERMION_OBJECT_INTEROP.md).

The ten-minute workflow is conversion followed by the unchanged native consumer:

```python
import torch
import openfermion
import flagquantum as fq
from flagquantum.ecosystem.openfermion import from_openfermion_fermion
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable

old = openfermion.FermionOperator("0^ 0", 1.5)
native = from_openfermion_fermion(old)
encoded = encode_fermion(native, n_modes=2, encoding="jordan_wigner")
circuit = fq.Circuit(2).x(0)
value = fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))])
assert abs(float(torch.as_tensor(value.expectation()).reshape(-1)[0]) - 1.5) < 1e-6
```

Change the source coefficient and rerun this path, then run
`python -m pytest tests/unit/test_openfermion_conversion.py
 tests/integration/test_openfermion_interop.py -q` (as one shell command).
Tests exercise real upstream operators, independent matrices, round trips,
contractions, exact small coefficients and malformed mutable terms. Without
OpenFermion the upstream tests skip; the fresh-interpreter native isolation test
still executes. Conversion preserves mode labels; neither algebra records idle
register width, so always pass it explicitly to native encoding or inspection.
Exported raw term dictionaries may differ from upstream normal ordering while
representing the same operator. Observable Hermiticity checks remain native.

For an existing `openfermion.InteractionOperator`, use
`from_openfermion_interaction(old)` and keep the same encoding and measurement
consumer above. This imports spin-orbital operator coefficients directly; it
adds no half-factor and does not interpret spatial chemistry integrals. Real or
complex numeric dense NumPy tensors must have shapes `(n, n)` and `(n, n, n, n)`.
The import reads all entries in O(n^4) time, including validating entries whose
ladder products vanish. It retains no arrays and preserves tiny nonzero
coefficients without a tolerance. This experimental import has no reverse tensor
export and does not construct a molecular problem. See the separate interaction
import extension in the contract for its exact formula and acceptance checks.

The optional lane's UCCSD replacement scenario checks that this imported value
and a native Hubbard Hamiltonian give matching energies, gradients and SGD
updates through the same unchanged consumer. Run
`python -m pytest tests/unit/test_openfermion_conversion.py -k replacement -q`
after installing the optional upstream environment. This is a small CPU/full-JW
training proof; parameter differentiation stays in native circuits rather than
in the imported static coefficients.
