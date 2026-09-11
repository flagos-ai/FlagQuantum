# Ecosystem interoperability

`flagquantum.ecosystem` is the interoperability boundary. It translates objects
from optional external frameworks into FlagQuantum-owned `CircuitIR` and
translates owned programs back only when explicitly requested. Conversion
reports make every unsupported or lossy semantic difference visible.

Interop does not define a second canonical IR, compile programs, select or
execute backends, implement numerical kernels, or let framework objects enter
Core, Compiler, Runtime, or Simulation. Importing this package and listing its
adapters must not import an optional framework.

## Where to start

- `contracts.py`: framework-neutral adapter and conversion-report protocol.
- `registry.py`: immutable, lazy adapter discovery.
- `conformance.py`: common round-trip and rejection checks.
- `<framework>/conversion.py`: external object to/from `CircuitIR` conversion.
- `<framework>/models.py`: adapter-local results and errors.
- `<framework>/adapter.py`: the small implementation of the common protocol.

`qiskit/execution.py` is recorded migration debt, not an example for new
adapters: external target execution belongs to Remote once the owning
contract is available. Do not add more execution behavior under Interop.

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
