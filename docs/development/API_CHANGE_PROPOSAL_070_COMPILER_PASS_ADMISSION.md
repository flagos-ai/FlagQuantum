# API Change Proposal 070: Give the compiler-pass extension kind a real crossing

## Status

**Proposed. No Stable Core change requested.**

This proposal connects two facilities that already exist in the repository: the
`compiler_pass` extension protocol, which has been declared and approved since
[`API_CHANGE_PROPOSAL_019`](API_CHANGE_PROPOSAL_019_COMPILER_EXTENSIONS.md), and
the circuit optimization pipeline, which is a fixed sequence of inline calls. It
adds no new plugin concept, no new extension kind, no new entry-point group, and
no new manifest format.

## Problem

### The recorded condition that withdrew the pass protocol has fired

[`FQ-PASS-MANAGER-CONTRACT-20261001.md`](../api-changes/FQ-PASS-MANAGER-CONTRACT-20261001.md)
proposed **not** publishing a pass manager in M1, and made that withdrawal
contingent on three triggers. Its trigger 1 is:

> A pass must be installed without editing the repository -- a plugin, an
> out-of-tree backend, or an ecosystem adapter that needs its own transformation.

That trigger has fired, and the capability-maturity and parity records are the
evidence: `compiler_pass` is one of the nine declared extension kinds, and the
`compiler_plugin_ecosystem` parity row is `partial` for exactly this reason. The
withdrawal is therefore no longer available, and the same document states what the
reopening owes: "the proposal should be the narrow Alternative 3, and it must state
how pass evidence survives composition." Sections 1, 2, and 6 answer that.

This proposal is narrower than the document's Alternative 3. It publishes no
protocol: the registry and the manager are Compiler-internal, are not exported by
`flagquantum.compiler`, and add no name to the stable root surface. The extension
protocol it serves is the one already approved by
[`API_CHANGE_PROPOSAL_019`](API_CHANGE_PROPOSAL_019_COMPILER_EXTENSIONS.md), which
is the existing authority for a plugin-supplied compiler transformation.

### The gap the extension kind exposes

The extension SDK declares nine kinds, and `compiler_pass` is one of them.
`CompilerPassExtension` fixes the member such a pass must implement:
`transform(self, program) -> program`, and the SDK negotiates, version-checks,
activates, and isolates an extension that implements it. The manifest kind
validates, and the entry-point group
(`flagquantum.extensions`, name `compiler_pass.<manifest-name>`) resolves.

Nothing consumes it. `flagquantum/ecosystem/extensions/sdk.py` exposes the
protocol, and no module in the repository ever asks for one. The compiler-side
reason is visible in the optimization path itself: the pipeline was nine inline
calls inside `flagquantum/compiler/pipeline.py`:

```python
while True:
    previous = len(ir)
    ir = remove_zero_state_resets(ir)
    ir = remove_diagonal_gates_before_measure(ir)
    ...
    if len(ir) == previous:
        break
```

That spelling names no pass. There is nothing to replace, nothing to list, and
nothing to supply from outside the repository, because a pass is not an object
here -- it is a line of code in the middle of a function. The consequences are
concrete:

- A pass cannot be swapped. Replacing one means editing `pipeline.py`.
- A pass cannot be resolved by name, so no sequence can be composed, printed, or
  tested as a sequence.
- A plugin has no name to register under, so the `compiler_pass` kind is a
  declared capability with no consumer. The repository's own capability-maturity
  and parity records describe it as designed-but-unreachable, which is accurate.

The `compiler` kind does not have this problem, because
`compile_with_extension` already carries it into `fq.compile(compiler=...)`.
That path transforms a whole program. A pass is the smaller hook, and it is the
one that is missing.

## Decision

### 1. The pipeline becomes a named sequence, resolved through a registry

`flagquantum/compiler/pass_manager.py` is new and owns three things: a pass name,
an ordered sequence of those names, and the manager that runs one over a program.
`OPTIMIZATION_PIPELINE` is the existing nine-step round, in the existing order,
with no step added or removed. `PassRegistry` is immutable; adding a pass returns
a new registry and mutating the mapping raises. `PassManager.run` applies a
sequence; `PassManager.to_fixed_point` applies it until a round removes no
instruction, which is the condition the inline loop tested.

The module owns no pass body. Every built-in implementation stays where it
already is, and `pipeline.BUILTIN_PASSES` binds the names to those functions. The
module also owns no extension protocol: it accepts a callable.

### 2. The replacement is the proof, not the interface

`_optimize_to_fixed_point` is deleted and re-expressed as
`PassManager(default_pass_registry()).to_fixed_point(ir, OPTIMIZATION_PIPELINE)`.
`optimize` and `compile` are not modified in any other way: same signature, same
result, same callers. This satisfies the replacement gate in
[ARCH-012](../architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md):
one implementation is swapped behind an existing boundary without modifying its
consumers. The round limit was `len(ir) + 1` inline and is `len(ir) + 1` by
default, so the loop terminates at the same point it did before.

### 3. Built-in pass names are reserved

The names `compile` is defined by may not be redefined. `PassRegistry` refuses to
register a name that is already present, and refuses a built-in name even on an
otherwise empty registry. An extension cannot reach a built-in name by
construction either, because the host builds the registered name as
`extension.<manifest-name>.<declared pass_name>`. Two independent checks, neither
relying on the other.

### 4. One admission crossing, in the ecosystem package

`flagquantum/ecosystem/extensions/pass_admission.py` is the crossing, and it is
the sibling of `admission.py` rather than a modification of it. It validates the
`pass_name` declaration, negotiates `circuit_ir`, performs the SDK-version check
through `ExtensionRegistry` rather than restating it, starts the handle, and
returns a registry that resolves the admitted pass alongside the built-ins.

It creates no registry, manifest format, protocol, entry-point group, or
discovery path. It is not added to
`flagquantum.ecosystem.extensions.__all__`, which
[`contracts/extension-protocol-v1-candidate.json`](../../contracts/extension-protocol-v1-candidate.json)
pins as an exact set; that approved contract is unmodified.

The registered route closes over the `ExtensionHandle`, not the extension object,
so a pass invoked after its extension closed fails closed rather than running
through a lifecycle that has already ended.

### 5. The default path stays built-in only

`fq.compile` and `fq.optimize` resolve built-in names. An admitted pass is not
added to the default registry: a plugin that could silently join the pipeline
`compile` is defined by would be exactly the hazard the reserved names guard
against, and would make a plugin installation change the meaning of a
Stable-Core-adjacent entry point with no visible cause. A host that wants an
out-of-tree pass names it explicitly, through `optimize_with_extension_pass` or a
`PassManager` it builds itself.

### 6. How pass evidence survives composition

The reopening condition asks this explicitly, and the answer is bounded on purpose.

What survives:

- **The order is data, not control flow.** `OPTIMIZATION_PIPELINE` is a tuple, so
  the sequence a compilation ran is inspectable, printable, and diffable without
  reading a function body, which was not true of nine inline calls.
- **A pass is identified by name at the moment it misbehaves.** The manager checks
  every result and raises `CompilationError` naming the pass, so a plugin that
  returns something other than FlagQuantum IR produces a diagnostic that names the
  plugin, not a type error attributed to the loop.
- **A built-in name resolves to the function that owns the transformation.** The
  registry tests pin every binding by identity against its own module, so the
  registry cannot silently drift away from the pass it claims to name.
- **An admitted pass is authored in and traced to one extension.** The registered
  name is host-built as `extension.<manifest-name>.<declared pass_name>`, so the
  name that appears in a diagnostic is attributable to a supplying extension.

What does not survive, and is recorded here rather than left implicit:

- **No per-pass provenance reaches the result.** `optimize` and `compile` return
  `CircuitIR`, which carries no record of which pass changed what. The only
  pass-record type in the repository is `PassRecord` in the private hybrid layer,
  and the canonicalization path does not emit one. The registry-plus-manager makes
  the order nameable and checkable; it does not add an evidence trail to the
  optimized program, and this proposal does not claim one.
- **No sequence is checkpointed or resumable.** A pass either runs to completion
  inside a round or the round raises; there is no partial-program artifact between
  passes.

The four surviving properties are exactly what the plugin use case needs, and the
two gaps are the ones no current consumer has asked for. Adding either would be a
new contract of its own, and neither is in this change.

## Public API

No public API changes. `flagquantum.compiler.pass_manager` and
`flagquantum.ecosystem.extensions.pass_admission` are both importable modules
outside the stable root surface; the stable root exports, their signatures,
defaults, and result fields are untouched.

```python
# A host admits one third-party pass and compiles with it. The built-in pipeline
# runs first and the admitted pass runs once after it.
from flagquantum.ecosystem.extensions.pass_admission import (
    optimize_with_extension_pass,
)

optimized = optimize_with_extension_pass(ir, extension=create_extension())
```

```python
# A host inspects the sequence instead of running it, which was not possible when
# the pipeline was nine inline calls.
from flagquantum.compiler.pass_manager import OPTIMIZATION_PIPELINE

for name in OPTIMIZATION_PIPELINE:
    print(name)
```

## Compatibility

Additive. No existing name is removed, renamed, or given a new default, and no
serialized schema changes. The two ecosystem modules are siblings, so an
installation that uses backend admission is unaffected.

## Acceptance

- The built-in pipeline is bit-identical for every input: same names, same order,
  same round limit. `tests/unit/test_compiler_pass_manager.py` pins the order and
  the two-round case the inline comment named.
- Each built-in registry binding agrees with the module that implements the pass,
  so the registry cannot drift away from the pass that owns the transformation.
- An extension pass changes compilation with no FlagQuantum source modified and
  no name added to any list in the Compiler.
- No extension value can produce a registered name equal to a built-in name.
- Every failure mode is rejected before the pass is registered: an unusable
  declaration, a non-`compiler_pass` kind, a missing `transform`, a refused
  negotiation, an extension that declares no `circuit_ir` support, an SDK
  mismatch, and a failed start.
- The extension handle is closed even when the pass fails.
- `tests/unit/test_compiler_pass_admission.py` is the scenario and failure-mode
  suite; `tests/unit/test_compiler_pass_admission_contract.py` checks this
  contract against the code.

## Non-goals

- Analysis passes. CUDA-Q separates analysis from transformation; this adds no
  analysis concept, because no FlagQuantum consumer needs one yet.
- A public `fq.PassManager`. The registry and manager are Compiler-internal.
- A pass-level conformance runner. `run_compiler_conformance` covers the
  `compiler` kind.
- A C ABI for compiler passes. That is a separate contract and must not be built
  before a real plugin demands it.
- Per-compilation discovery, negotiation, or version checks.
