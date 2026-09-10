# Public typing corrections for Circuit, execution scope, and CPU probes

Status: proposed; API-owner approval pending. Production code is unchanged.

## Problem and scope

Five strict-mypy diagnostics arise from incomplete or inaccurate public type
contracts. Users need accurate IDE results for Circuit inspection and noise
execution, exception propagation from strict execution scopes, and support for
immutable CPU probe metadata. This proposal changes three files and introduces
no exported names, dependencies, result fields, numerical behavior, or schemas.

Owners: Integration/API owner, Runtime, and Compute. No approval is recorded.

## Exact changes

In `flagquantum/circuit.py`:

```python
# Before
def noisy_density_matrix(self, noise_model=None) -> torch.Tensor: ...
def analysis(self): ...
def runtime_plan(self, **options: Any): ...

# After
def noisy_density_matrix(
    self, noise_model: NoiseModel | None = None
) -> torch.Tensor: ...
def analysis(self) -> CircuitAnalysis: ...
def runtime_plan(self, **options: Any) -> RuntimeSelectionPlan: ...
```

The delegate already accepts `NoiseModel | None`. `analysis` delegates to
`runtime.planner.analyze`, which returns the existing
`runtime.execution_plan.CircuitAnalysis`. `runtime_plan` delegates to
`runtime.planner.plan_runtime_selection`, which returns the existing
`runtime.planner.models.RuntimeSelectionPlan`. Add only TYPE_CHECKING imports
for the two return classes; do not move their ownership or eagerly import the
planner. The existing optional noise type import is reused.

In `flagquantum/runtime/routing.py`:

```python
# Before
def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool: ...

# After
def __exit__(
    self, exc_type: object, exc: object, traceback: object
) -> Literal[False]: ...
```

Import `Literal` from typing. Preserve the existing `return False` body:
exceptions must continue to propagate out of `StrictExecutionScope`.

In `flagquantum/compute/cpu_target_capabilities.py`, replace each of the six
mutable protocol attributes with a read-only property returning `str`:

```python
class CPUCapabilityProbe(Protocol):
    @property
    def target_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    @property
    def target_revision(self) -> str: ...

    @property
    def environment_id(self) -> str: ...

    @property
    def source_ref(self) -> str: ...

    @property
    def target_class_source_ref(self) -> str: ...

    def observe(self) -> CPUCapabilityObservation: ...
```

The adapter reads these fields and never writes them. The existing frozen
`_ObservedCPUProbe` must remain frozen. Mutable provider instances can still
satisfy the read-only protocol structurally. Do not add setters or mutation to
the observation path.

## Compatibility and migration

Call syntax, defaults, return objects, JSON bytes, exception classes, and
execution behavior remain unchanged. Runtime annotation inspection sees the
new types. Applications reading probe metadata need no source migration.
Code assigning metadata through a `CPUCapabilityProbe` annotation will become
a static error: perform such updates through the provider's concrete mutable
type before passing the probe to the adapter. This static compatibility impact
requires explicit approval; it is not dismissed as formatting.

Example unchanged usage:

```python
circuit = fq.Circuit(2).h(0).cx(0, 1)
analysis = circuit.analysis()
selection = circuit.runtime_plan()
density = circuit.noisy_density_matrix()
```

First deprecation version: none proposed. Removal version: none proposed.
No callable or runtime behavior is removed. If the API owner requires a
deprecation window for protocol writes, defer that protocol change separately.

Alternatives rejected: make the frozen probe mutable, obscure the constant
False return, use casts/ignores, or introduce replacement result abstractions.
These would hide the contract mismatch or weaken existing guarantees.

## Verification and acceptance after approval

1. Check unchanged Circuit results against the existing delegate results,
   including noise=None and an explicit NoiseModel.
2. Verify a sentinel exception escapes StrictExecutionScope unchanged.
3. Verify frozen and mutable probe implementations produce equivalent capability
   snapshots; include static conformance for both implementations.
4. Run relevant Circuit/planner/noise tests, strict-execution tests, and
   `tests/team/compute/test_cpu_target_capabilities.py`.
5. Run strict mypy over the complete package with Python 3.12. This proposal
   targets five diagnostics; the expected reduction is not a measured result.
6. Run Ruff, Black, architecture and public API checks. Review each signature
   difference against this proposal. Do not regenerate snapshots wholesale.
7. Add a release note describing the improved annotations and the read-only
   protocol contract. Document the verified outcome in the remediation log.

Other parameter-binding, instruction, execution-plan, optimizer-constructor,
and Triton typing issues are outside this approval request.
