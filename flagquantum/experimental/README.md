# Experimental

Hold the explicitly unstable surface of FlagQuantum: APIs that may change or
disappear in any release, with no compatibility guarantee. `__init__.py` states
that contract and does not re-export feature symbols, so discovery stays small
and ownership stays clear.

This domain owns the unstable **re-export layer and the artifact codec**. It does
not own the implementations it publishes: every behavioural symbol here is
imported on first access from the domain that owns it, so each capability keeps a
single authoritative implementation rather than a copy.

Two things set this package apart from the other target domains. First,
`flagquantum/experimental/**` is not assigned to any team in
`team-ownership.toml`, so `check_team_scope.py` rejects a domain team here;
treat changes as integration changes. Second, the three reserved modules
(`planning`, `execution`, `mps`) are deliberate empty markers that reserve a
namespace and record where the supported equivalent lives. They must not grow
implementations.

`contracts/public-api-v0.2-baseline.json` records the `flagquantum.experimental`
**module** as a stable path, but none of the names advertised underneath it: the
baseline lists the package, not its unstable contents. Keep it that way. A name
reached through this package stays unstable by design, and promoting one to the
Stable Core surface is an API change proposal, not an edit here. Run
`python tools/check_capability_maturity.py` before adding a name.

## What guards this surface

`tests/api_contract/test_experimental_namespace_contract.py` is the contract test
for this package, and it pins more than a re-export facade usually gets. It reads
`contracts/experimental-surface-v2-candidate.json` and asserts, for all nine
namespaces, that `__all__` equals the contract's `discoverable_exports` list
exactly and that every listed name resolves. It also asserts that internal records
(`DynamicExecutionResult`, `TEBDResult`, `JAXStatevectorShardState`, …) are not
discoverable, that every route removed in v1 now raises `AttributeError`, that flat
feature routes such as `run_tebd` are rejected at the root with a "domain
namespace" message, and that the reorganization did not change `fq.__all__`.

Measured teeth: renaming `run_dynamic` to a private name in its owning module
`flagquantum/runtime/dynamic/execution.py` makes
`test_each_experimental_namespace_matches_machine_contract` fail; restoring the
name makes the file pass again (6 passed).

So `__all__` here is not documentation, it is a pinned contract. Adding, removing,
reordering, or re-pointing a name requires editing
`contracts/experimental-surface-v2-candidate.json` in the same change. That
contract and this package are both integration surfaces, which is the main reason
the table below is worth keeping current.

The convergence test `tests/unit/test_api_namespace_convergence.py` is a separate
guard: it covers the stable migration destinations, not this package.

## Choose an entry point

| Module | Advertised names | Owner of the implementation |
| --- | --- | --- |
| [`artifacts.py`](artifacts.py) | `ProgramArtifact`, `CompilationEvidence`, `dump_program_artifact`, `load_program_artifact`, `dump_compilation_evidence`, `load_compilation_evidence` | **This module.** The only one that owns behaviour; it is the read-only view over versioned artifact envelopes. |
| [`distributed.py`](distributed.py) | `train_distributed_statevector`, `train_distributed_mps` | `flagquantum.runtime.executors.statevector.training`, `…mps.training` |
| [`dynamic.py`](dynamic.py) | `run_dynamic`, `assess_dynamic_backend` | `flagquantum.runtime.dynamic.execution`, `…deployment` |
| [`numerics.py`](numerics.py) | `execute_split_real_imag_expectation`, `execute_split_real_imag_statevector`, `parameter_shift_split_real_imag_gradient` | `flagquantum.runtime.executors.statevector.split_real_imag` |
| [`simulation.py`](simulation.py) | `run_tebd` | `flagquantum.simulation.mps.tebd` |
| [`interop.py`](interop.py) | `braket`, `cirq`, `cudaq`, `pennylane`, `qiskit` | `flagquantum.ecosystem.<name>` — adapters, not feature functions |
| [`planning.py`](planning.py) | *(none)* | Reserved; use `flagquantum.plan` |
| [`execution.py`](execution.py) | *(none)* | Reserved; no public entry points |
| [`mps.py`](mps.py) | *(none)* | Reserved; no public entry points |

Everything here is CPU-only and offline: no function in this package contacts a
target or requires credentials.

## How delegation works

Each delegating module resolves names lazily, so importing
`flagquantum.experimental` does not import the runtime, the ecosystem adapters, or
TEBD:

```python
def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("flagquantum.runtime.dynamic"), name)
    raise AttributeError(name)
```

Two consequences matter when changing this domain:

- A name that moves in its owning module is caught by the contract test only
  because `discoverable_exports` resolves it. Rename the owner without updating the
  delegating module and the test fails, which is the intended outcome — but note
  that this makes an owning-domain rename a two-domain change.
- `__all__` is pinned, not merely documented: the contract test compares the list,
  so reordering names fails even though behaviour is unchanged.

## Shortest change path

To add, move, or remove an unstable name:

1. Change the owning domain first (for example
   `flagquantum/runtime/executors/statevector/split_real_imag.py`); that domain's
   own conventions and reviewers apply.
2. Update `__all__` in the matching `flagquantum/experimental/<domain>.py`, the
   `import_module` target if the owner changed, **and**
   `contracts/experimental-surface-v2-candidate.json`. The contract test compares
   the list exactly, so a missed edit fails rather than drifting silently.
3. Run the checks from the repository root:

   ```bash
   python -m pytest tests/api_contract/test_experimental_namespace_contract.py -q
   python tools/check_capability_maturity.py
   python tools/ci_tier.py pr-default
   ```

Do not add an implementation to this package. If a capability is stable enough
to have a real implementation, it belongs in its owning domain; if it is not,
it belongs here only as a delegation entry.
