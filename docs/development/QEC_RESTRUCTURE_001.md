# QEC-RESTRUCTURE-001: package audit and migration proposal

Status: **plan and compatibility strategy approved by the user on 2026-10-23;
QEC-RESTRUCTURE-002 authorized as a separate PR**.
Audit base: `ff69b2526809772b7fadde5bcb420f9f5899dcf6` (`origin/main`, fetched
2026-10-23). Branch: `refactor/qec-restructure-001`. Owner: Core team;
serialization and public compatibility require API-maintainer review.
Classification: `single_device_fast_path`; no distributed or performance claim.

This change adds documentation, compatibility fixtures, tests, and an executable
example only. It changes no `flagquantum/qec` implementation, package layout,
registry, dependency manifest, or Stable Core contract. DynamicCircuit PR #731
is outside its scope: do not modify, merge, or use its branch as a dependency.
Implementation begins with the separately reviewed
[QEC-RESTRUCTURE-002](QEC_RESTRUCTURE_002.md). The audit evidence below remains
the original 001 record; its compatibility fixtures must not be refreshed.

## Evidence and existing surface

The package contains **33 Python modules, 16,452 lines, and 108 ordered top-level
exports**. The [AST inventory](qec_restructure_001_inventory.json) records every
module's definitions and imports, including local and type-checking imports;
these are potential edges, not a claim that every import is eager. It also lists
83 textual consumers outside the package. The consumer list covers tracked files at the audit base and excludes the new
artifacts in this PR.

The [compatibility fixture](../../tests/qec/fixtures/restructure_001_api.json)
records all 108 exports, callable signatures (excluding Python-owned Enum
construction), public Python method signatures,
properties, dataclass field order/types/defaults/factories/frozen state, enum
values, defining import paths, registry names, and trusted protocol-4 pickles.
It also covers declared module exports and locally defined public classes and
functions beyond the root facade, plus a CPU seeded DEM sample and exact rates.
It is a migration baseline, not a new declaration that experimental QEC belongs
to Stable Core. Do not regenerate it to make a migration pass. Existing Stable
Core snapshots and policies remain authoritative and unchanged.

| Current modules (all names below end in `.py`) | Current responsibility and proposed home |
| --- | --- |
| `__init__` | Ordered public facade; remains `qec/__init__.py` |
| `pauli`, `gf2`, `logical` | Algebra and logical certification; retain small shared leaf modules |
| `types` | Repetition records; retain shared records until a separate justified split |
| `codes` | Protocol, checks, repetition/Steane/rotated code classes, CSS helpers, colour/toric factories, code registry and lazy CSS export; split under `codes/` |
| `css_code`, `qldpc`, `bicycle`, `reed_muller`, `subsystem`, `floquet` | Code records/factories; `codes/` |
| `surface` | Independent rotated class and ZXXZ code/lattice; `codes/surface/` |
| `circuit` | Memory declaration, source generation, measurement/detector/observable layout; `circuits/memory.py` |
| `dem` | DEM records, merges, sampling, rates, construction facade; `dem/model.py` |
| `dem_text`, `dem_alternatives` | Text parser and mutually exclusive mechanisms; `dem/text.py`, `dem/alternatives.py` |
| `dem_matrices` | CSS matrix record, validation/extraction, noise-to-mechanism conversion; split record into `codes/matrices.py` and conversion into `dem/matrices.py` |
| `dem_construction` | Forced-signature execution, gate-noise construction and matrix re-exports; separate `execution/signatures.py` from `dem/construction.py` |
| `dem_circuit` | DEM ↔ IR realization; `dem/circuit.py` |
| `chunks` | DEM chunk/spec/layout/seam stitching and closing; `dem/chunks.py` |
| `decoding_graph` | DEM-to-weighted-graph representation; `dem/graph.py` |
| `decoders` | Repetition history/streaming protocols and lookup/temporal implementations; `decoders/repetition.py` |
| `matching`, `belief_propagation`, `bposd`, `sliding_window` | In-tree decoding algorithms; `decoders/` |
| `registry` | DEM decoder protocols and sole name registry; `decoders/registry.py` |
| `context` | Memory-to-DEM/measurement map composition; `execution/context.py` |
| `repetition`, `sampling` | Hybrid feedback execution and stabilizer sampling; `execution/` |
| `noise` | Phenomenological/profile records plus executing sweep; retain records at `qec/noise.py`, extract sweep to `execution/noise_sweep.py` |
| `adapters` | Optional PyMatching dependency boundary; `adapters/pymatching.py` |

## Dependencies and coupling that constrain the split

- `pauli` and `gf2` use the standard library. Code protocols/checks depend on
  these leaves. Code factories share private matrix/distance helpers in `codes`;
  `qldpc` also supplies helpers to bicycle and Reed-Muller construction.
- `css_code` currently imports `CssCodeMatrices` through `dem_construction`.
  That re-exports the record from `dem_matrices`, which imports `noise`, whose
  sweep imports `repetition` and thus compiler/runtime. A matrix-only code
  construction can therefore reach execution infrastructure through imports.
- `circuit` consumes codes, logical certification and Pauli. `MemoryCircuit`
  methods locally import compiler/IR and DEM; moving the module alone does not
  establish a dependency boundary. Retain public methods, delegating their
  existing operations through narrow local imports when the approved split lands.
- `dem` imports `dem_construction` and `noise`. Construction directly imports
  `compiler._hybrid`, core IR/operator schema, noise/Kraus channels and
  `runtime.dynamic.hybrid_session`. `dem_circuit` also consumes core IR.
- `sampling` uses core IR, noise and `simulation.stabilizer`;
  `repetition` uses compiler capture/lowering and runtime feedback/session APIs.
  These are existing seams, not authorization to change compiler/runtime or #731.
- Decoders consume DEM/graphs and errors; BP/OSD matrix arithmetic uses PyTorch.
  `context` composes DEM, memory declarations and measurement maps. Keep it above
  those inputs rather than making code definitions depend on decoder context.
- Core runtime dependency is PyTorch (`>=2.10,<2.15` in the manifest). Stim
  (`>=1.16,<2`) and PyMatching (`>=2.3,<3`) are optional extras already declared.
  Text DEM parsing does not require Stim. Importing QEC must not import
  PyMatching; requesting its registered name without the extra must retain
  `MatchingDependencyError`. No new dependency is proposed.

### Identity and registration findings

1. **Two distinct rotated surface classes already exist.** The root export and
   `get_code("rotated_surface")` use `codes.RotatedSurfaceCode`;
   `surface.RotatedSurfaceCode` is another dataclass used by ZXXZ and existing
   tests. Equal-distance objects have equivalent checks but unequal dataclass
   identity. Preserve both identities during migration. Sharing lattice helper
   implementation can be reviewed later; collapsing classes is a separate
   compatibility decision, not an incidental cleanup.
2. `codes.__getattr__` lazily exposes `CssCode` to avoid a cycle; `__dir__` and
   `codes.__all__` also advertise it. Preserve these old-path behaviors.
3. Code registration currently names exactly `repetition`, `rotated_surface`,
   `steane`. Other factories are directly available, not missing registrations
   to be added during a structural refactor. Preserve sorted names, validation,
   constructor option handling, duplicates, and explicit replacement.
4. DEM registry names are `belief_propagation`, `belief_propagation_osd`,
   `minimum_weight_matching`, `pymatching`, `sliding_window_matching`.
   The repetition history decoders intentionally use another protocol and are
   not members. Preserve graph-vs-model/text routes and BP+OSD graph refusal.
5. `sliding_window` imports `registry.get_decoder`; the registry registers it
   lazily through `_register_windowed_decoder`. Every access route must retain
   this behavior and must not overwrite an explicit plugin replacement. There
   must be one authoritative mutable table per registry, never a copied table
   in an old-path compatibility module.

## Proposed directory boundaries (not implemented)

```text
qec/
  __init__.py             # unchanged top-level facade
  pauli.py, gf2.py, logical.py, types.py, noise.py
  codes/
    __init__.py           # preserves former qec.codes exports/lazy CSS lookup
    checks.py, registry.py, repetition.py, steane.py, css.py, matrices.py
    colour.py, toric.py, qldpc.py, bicycle.py, reed_muller.py
    subsystem.py, floquet.py
    surface/
      __init__.py, lattice.py, rotated.py, _rotated_frame.py, zxxz.py
  circuits/
    __init__.py, memory.py
  dem/
    __init__.py           # preserves former qec.dem exports
    model.py, text.py, alternatives.py, matrices.py, construction.py
    circuit.py, chunks.py, graph.py
  decoders/
    __init__.py           # preserves former qec.decoders exports
    repetition.py, matching.py, belief_propagation.py, bposd.py
    sliding_window.py, registry.py
  execution/
    __init__.py, repetition.py, sampling.py, signatures.py, context.py
    noise_sweep.py
  adapters/
    __init__.py           # preserves former qec.adapters exports
    pymatching.py
  # Existing non-colliding flat paths remain compatibility import modules.
```

This tree describes responsibility, not an instruction to create every file at
once. Avoid forwarding-only submodules beyond compatibility requirements.
`rotated.py` retains the identity published by the old `codes` module;
`_rotated_frame.py` retains the distinct identity historically defined in
`surface.py` and used by ZXXZ. The old flat `surface` facade must continue to
resolve that second class. The surface package facade can expose the existing
root class without conflating the two. Shared lattice helpers do not authorize
collapsing their dataclass identities.

Each migrated directory needs a short README covering ownership, forbidden
responsibilities, permitted dependencies, public entries and a ten-minute example.

Desired direction: algebra/records → codes → circuit declarations → DEM
construction/graphs → decoders; execution composes these with compiler/runtime
and simulation. Adapters depend on QEC-owned contracts and the external SDK at
call time. The arrows describe consumers building on preceding responsibilities.
`CssCodeMatrices` must become a code-owned record before moving code definitions;
noise records must stop pulling in a sweep's executor. Code-local matrix
validation and bounded-distance helpers move with their code consumers, without
algorithm changes; DEM fault expansion must not be imported by the matrix record. No new registry, protocol,
IR, manager, dependency or architecture-policy exception is proposed. DEM factory
methods and memory lowering methods retain their signatures and use narrow local
composition imports; the pure data modules must not eagerly import execution.

## Compatibility strategy and risks

| Surface | Risk | Required handling and evidence |
| --- | --- | --- |
| Root imports | Export order, signatures, field defaults, enums or class identity drift | Preserve all 108 exports and existing objects; baseline plus scenario tests |
| Old module paths | `codes.py`, `dem.py`, `decoders.py`, `adapters.py` collide with package names | Replace each atomically with a package `__init__`; never ship both module and directory; test wheel and sdist clean installs |
| Other flat paths | Existing direct imports, monkeypatch targets and module attributes disappear | Preserve re-exports, documented aliases and local `__all__`/`__dir__`; audit callers before moving private patch targets |
| Pickle and reflection | Re-export permits old unpickling but a moved class changes newly written pickle globals, `__module__`, repr and plugin lookup | Preserve defining metadata for existing public types/functions at historical paths, with old aliases resolving the exact object. Do not blindly rewrite `__module__` for every object. Review the two rotated classes separately; load retained old payloads and round-trip new payloads in a fresh process |
| Structured serialization | DEM text, binary records, IR dictionaries or identities change indirectly | Preserve schema/order/dtype/shape, source generation, probability precision, error IDs, mutually exclusive mechanisms and seed consumption. Existing text/binary/round-trip tests are part of the gate |
| Registries/plugins | Split tables, eager optional dependency, registration order or replacement loss | Preserve decorators and return identity, lazy registration, unknown-name/type errors, explicit replacement and source routes; test mixed old/new imports and third-party subclasses |
| Documentation | Sphinx/autodoc object paths and guide examples resolve differently | Keep root documented entry points; update physical source links only, retain historical API cross references, run guides/examples and link/doc checks; new paths must not be advertised before implemented |
| Numerics/execution | Reordering error mechanisms, RNG calls or layout construction changes results | Exact deterministic records/rates, matrices and seeded same-environment output; existing independent Stim/PyMatching statistical tolerances unchanged |
| Packaging | Working-tree tests hide absent package data/shims | Build wheel/sdist, install each into clean environments, execute old imports, old-pickle load and golden path outside checkout |
| External plugins | Repository search cannot enumerate downstream dynamic imports, custom picklers or monkeypatches | No plugin entry-point discovery was found in QEC; decorators are the extension seam. Obtain maintainer/downstream feedback and exercise a plugin fixture; do not claim ecosystem-wide certification |

Compatibility owner: Core team with API-maintainer review. Scope: historical
QEC paths and behavior at this base. Removal condition: explicit approved
migration/deprecation decision plus downstream evidence; no removal version is
currently approved, and this refactor has no removal target release. Removing
these aliases is a separate proposal. They must not silently expire after a
structural migration. A custom pickle reducer or serialized schema change would
likewise require its own approval.

The checked-in protocol-4 payloads cover 13 representative records/classes,
including both rotated surface identities. They are trusted test fixtures only.
They are not complete checkpoint support: tensor-backed records, mapping proxies,
custom plugins and all Python-version combinations still need targeted migration
validation. QEC currently defines no general `to_dict/from_dict` or checkpoint
schema; IR serialization and DEM text/binary formats remain separate contracts.
`torch.save` may embed these same Python globals; caller-owned
`torch.load(weights_only=True)` safe-global allowlists can also depend on those
paths. Preserve historical identity rather than silently extending an allowlist
or adding a reducer. No in-tree general cloudpickle/joblib/plugin checkpoint
contract was found, so those consumers require downstream evidence.

## Proposed PR batches and approval gate

| Batch | Scope after approval | Risk and acceptance emphasis |
| --- | --- | --- |
| 001 (this PR) | Audit, proposal, compatibility baseline, executable current workflow | No production changes; baseline failures reported without altering implementation |
| 002 | Atomically convert `codes.py` to `codes/__init__.py` with its bodies intact; separate noise records from executing sweep and extract the CSS matrix record into `codes/matrices.py`, keeping old exports | High package-collision/cycle risk; code-only construction, noise identity, matrices, old paths and import-order checks |
| 003 | Split the established `codes` package into code families and the surface subpackage; keep two rotated identities and code registry authority | High identity/pickle and geometry risk; every code family's checks/logicals/distances, old payloads, plugin factory tests |
| 004 | Move circuit declarations and execution composition, including forced signatures/context/sampling | High execution/RNG risk; exact source/layout/feedback baselines, matrix-vs-circuit semantics, deterministic and noisy sampling |
| 005 | Convert DEM to package; move parser/mechanisms/chunks/graphs and realization | High numerical/serialization risk; text precision, binary records, error IDs, seam composition and independent reference checks |
| 006 | Convert decoders/adapters to packages, move algorithms and single registries | High plugin/import-order risk; all carriers, optional-extra refusal/presence, sliding-window replacement, authority/cross-check parity |
| 007 | Documentation/package installation verification and removal of only demonstrably unreferenced private scaffolding | Medium packaging risk; wheel/sdist, API reference links, golden paths and contributor walkthrough |

Batch 002 creates the `codes` package atomically before adding its matrix
submodule; it must never leave a same-named module and package together. This
ordering avoids a temporary matrix home or a dependency on a package that does
not yet exist. Splitting the other code implementations follows in batch 003.

Batches may be split further for reviewability; do not combine behavioral fixes,
new code families, registry policy changes or DynamicCircuit work with moves.
Each batch is a separate reviewable PR with its runnable **existing public API**
example and exact expected output. Every batch must pass before the next one;
new main changes require a baseline comparison and conflict review, never a
snapshot refresh. New package paths become usable only in the PR implementing
them. PR #731 stays independent even if it later lands on main.

## Acceptance criteria

1. Approval explicitly covers the batch sequence and compatibility strategy;
   batch 001 approval alone does not authorize an API break.
2. All 108 ordered root exports, signatures/defaults, dataclass fields and
   historical defining identities remain unchanged. Old aliases and public
   protocols remain usable; no Stable Core snapshot is updated.
3. Code and decoder registry names, duplicates/replacement, lazy behavior,
   graph/model/text routing, rejection behavior and optional import timing match.
4. Existing deterministic code/circuit/DEM/decoder tests and seeded feedback
   results match; probability precision, merge/error-ID semantics, tensor
   dtype/shape and existing statistical tolerances are preserved.
5. Retained old pickle fixtures load; moved types round-trip across a fresh
   interpreter using old and new imports. Test supported Python 3.10–3.12 and
   install variants with neither extra, each extra and both extras in CI before
   declaring migration complete.
6. Focused QEC tests and the repository default smoke/unit tier pass; execution
   moves additionally pass local integration and independent Stim/PyMatching
   checks. Report skips and environment blockers accurately.
7. Clean wheel and sdist installations pass imports, pickle and golden-path
   checks outside the source tree; documentation links/autodoc resolve, each
   migrated domain has its README, and a contributor can explain a small change
   from that README and example.
8. Only approved paths change; no #731 edits/merge, new dependencies, registry,
   public abstraction, schema change, fallback or scalability claim.

## Runnable user example for this and subsequent PR descriptions

From an installed checkout (or with `PYTHONPATH=.`):

```bash
python examples/qec/restructure_compatibility.py
```

The [example](../../examples/qec/restructure_compatibility.py) uses only today's
public imports. It decodes a one-detector DEM, injects a middle-qubit error in the
second repetition round, and corrects it using an offline Pauli frame:

```python
from flagquantum.qec import (
    AUTHORITY_NAME, DetectorErrorModel, ErrorEvent, ErrorSchedule,
    get_decoder, run_repetition_memory_experiment,
)

model = DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
print("Correction observables:", get_decoder(AUTHORITY_NAME, model).decode((0,)).observables)
result = run_repetition_memory_experiment(
    error_schedule=ErrorSchedule((ErrorEvent(1, 1),)),
    rounds=3, shots=2, seed=53, strategy="trajectory",
    feedback_mode="offline_pauli_frame",
)
print("Logical error rate:", result.logical_error_rate)
print("Decoded data:", result.shot_records[0].decoded_data_bits)
```

Expected output:

```text
Correction observables: (0,)
Logical error rate: 0.0
Decoded data: (0, 0, 0)
```

No credentials, remote jobs, optional extra or accelerator are needed. This PR
protects the existing workflow; it does not make the proposed directory tree
available.

## Validation record

See the result record below; optional checks and broader tiers must be reported
separately from the new focused baseline. This is local compatibility evidence,
not a completed migration or ecosystem-wide compatibility certification.

Test environment: macOS arm64, Python 3.12.14, PyTorch 2.13.0, CPU. The reference
run additionally used Stim 1.16.0 and PyMatching 2.4.0 from a temporary dependency
directory; no repository dependency declaration or existing environment changed.

| Check | Result |
| --- | --- |
| New compatibility tests | 11 passed |
| Full QEC plus memory logical-error-rate reference tests, with both extras | 1,812 passed, 1 skipped (269.73 s) |
| Core-only QEC, memory reference and alignment mutation tests, before the last two baseline additions | 1,437 passed, 42 skipped (196.92 s); optional distributions absent |
| Fresh interpreters starting with `codes`, `registry`, or `surface` imports | All 13 historical pickle paths and both registries passed in each of 3 interpreters |
| User example, with and without extras | Exact three output lines shown above |
| QEC/CUDA-Q alignment checker | 493 checks passed |
| Core path-ownership check, Black and Ruff on the two added Python files | Passed |

The reference-run skip is the existing gate-noise case whose code has no `cx`
fault on which the selected channel can be placed. Core-only skips cover missing
optional references/samplers. Supported Python 3.10/3.11, separate one-extra
installs, wheel/sdist installation and downstream plugin checkpoints are migration
acceptance work, not certified by this audit. The smoke/unit result and its unchanged-base comparison follow.


The required default tier (`python -m pytest -m "smoke or unit" -q`) completed
with **14,726 passed, 22 failed, 6 errors, 278 skipped, 3,486 deselected** in
780.91 s. It is **not green**. All 28 failure/error node IDs were rerun in a
separate untouched worktree at the exact audit base, with the same interpreter,
extras and sandbox: **the same 22 failures and 6 errors reproduced** (16.77 s).
The inventory's `validation` section retains every node ID for review.

The reproduced groups are nine native CPU dispatch/initialization expectation
failures, seven Git evidence/provenance/history assertion failures, six local
network/preflight permission failures, and six Gloo process-group setup errors
(`Cannot resolve 127.0.0.1 to a (local) address`). These are outside QEC; this
audit does not change their implementations, weaken assertions, or treat them
as passed. A supported clean environment/main-baseline resolution is required
before claiming the default gate or a later migration fully green.
