# Public typing corrections for Circuit, execution scope, and CPU probes

Status: implemented and verified on 2026-09-10 under the recorded approval.

## Problem and scope

Five strict-mypy diagnostics arise from incomplete or inaccurate public type
contracts. Users need accurate IDE results for Circuit inspection and noise
execution, exception propagation from strict execution scopes, and support for
immutable CPU probe metadata. This proposal changes three files and introduces
no exported names, dependencies, result fields, numerical behavior, or schemas.

Owners: Integration/API owner, Runtime, and Compute. The user approved this
specific proposal with "do" after reviewing its scope and compatibility impact
on 2026-09-10. This approval covers the five changes below, including read-only
CPU probe metadata; it does not authorize unrelated API changes.

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

## Verification outcome

All five approved corrections are implemented. Full-package strict mypy with
Python 3.12 now reports 39 errors in 17 files across 445 source files, down from
44 errors in 20 files. The five targeted diagnostics are gone. This is not a
claim that the package is strict-clean.

The initial behavioral run passed 88 tests covering noise execution, strict
execution scopes, CPU capability snapshots, and five new contract tests. A
subsequent run passed the five new tests and two existing Circuit/planner tests
(7 passed in 0.90s). New tests cover original-exception identity, exact density
results with and without noise, delegate parity, and equal snapshots from
mutable and frozen CPU probes. The frozen probe remains immutable.

The new test file also passes strict mypy with imported-module diagnostics
silenced for this test-only check; the separate full-package check above retains
its original strict configuration. An initial test-only failure from dynamically
generated gate aliases was resolved using the existing typed `Circuit.gate`
method, without suppressing the diagnostic. Black formatting was corrected.

Ruff, Black, architecture, and the public API migration baseline pass. The
baseline required no update. The changed signatures are those listed above;
no snapshot was regenerated. Release notes record the static compatibility
impact. No GPU, QPU, or distributed capacity claims follow from this CPU work.

Logs: `/private/tmp/fq-approved-types-mypy.txt`,
`/private/tmp/fq-approved-types-focused.txt`,
`/private/tmp/fq-approved-types-focused-final.txt`, and
`/private/tmp/fq-approved-types-static-final.txt`.
