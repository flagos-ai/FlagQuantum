# API Change: `fq.from_openqasm` and the OpenQASM import subset

## Status

Implemented for review on 2026-10-02 with explicit API-owner authorization for
two additive Stable Core root exports, `fq.from_openqasm` and `fq.gradient`; this
document covers the first of the two. It adds one root name, changes no existing
signature, changes no serialized schema, and leaves `IR_VERSION` at `1.0`.

Under `docs/development/PUBLIC_API_PROTECTION.md` an additive stable API requires
a concrete user journey, a reason the name belongs in Stable Core rather than a
namespace, typing and documentation, executable behavior contracts, API-owner
approval, a release-note entry, and the updated machine-readable contract. Each
is recorded below.

## Problem

FlagQuantum can write OpenQASM but cannot read it. The emitter
`flagquantum.compiler.openqasm.emit_openqasm` reached Stable Core behaviour
without a Stable Core name, and no public entry point reads the text back:

```python
import flagquantum as fq
from flagquantum.compiler.openqasm import emit_openqasm

text = emit_openqasm(fq.Circuit(2).h(0).cx(0, 1))
```

A user who exports a circuit, stores the text, and later wants to run it again
must lower it by hand. The practical consequence is that a FlagQuantum program
has no round-trip path through the one interchange format the package already
writes, so exported text is a one-way artifact rather than a saved program.

The tree already contains an OpenQASM parser, but it is a *template-matching
conformance checker* rather than an importer: `_parse_openqasm` in
`flagquantum/compiler/target_conformance.py` requires a `template: CircuitIR`
argument and returns evidence about whether an emission matches that template.
It cannot read a program whose content is unknown in advance, so it is not a
substitute for this entry point.

The tests use the emitter's deep module path
(`flagquantum.compiler.openqasm.emit_openqasm`) because no `Circuit.to_openqasm`
method exists; the capability entry names that same path.

## Decision

### One new root export

`fq.from_openqasm(source: str)` reads the canonical subset FlagQuantum writes and
returns an `OpenQASMImport` with five attributes, two conversion methods, and one
re-emission method.

The return type is deliberately not a bare `Circuit`. `Circuit.from_ir` does not
carry `ir.measurements` across, and `fq.run` refuses a bare `CircuitIR` that
carries a terminal measurement with shots, requiring outputs to arrive as
explicit `OutputRequest` values. Returning a `Circuit` would therefore silently
drop the one part of an OpenQASM program a `Circuit` cannot represent. The
importer states what it read and lets the caller choose:

| Member | Meaning |
| --- | --- |
| `n_qubits` | Width of the declared quantum register |
| `version` | The declared OpenQASM version, `2.0` or `3.0` |
| `instructions` | The imported program, in source order |
| `measurement_qubits` | The qubit behind each classical bit, in bit order |
| `source` | The exact text the program was imported from |
| `n_wires` | Deprecated alias of `n_qubits`; warns on use |
| `to_ir()` | Canonical `CircuitIR`, carrying the measurement node |
| `to_circuit()` | `fq.Circuit` for the stable `fq.run` sampling path |
| `to_openqasm()` | Canonical text, in the version the source declared |

### The accepted subset is closed, and refusal is the failure mode

Import accepts only what the emitter writes, and it verifies that claim instead
of asserting it: after parsing, the importer rebuilds the IR, re-emits it, and
requires the result to be the same program statement for statement. Text that
means something the emitter would write differently is refused as
`not_canonical_text` rather than imported approximately. Comments, blank lines,
and spacing around punctuation carry no meaning and are not a refusal.

Eleven refusal reasons are a closed vocabulary, and each is reachable by a test:

```
duplicate_measurement_target  empty_source            incomplete_measurement
invalid_parameters            malformed_statement     not_canonical_text
out_of_range_qubit            unbound_parameter       unknown_gate
unknown_register              unsupported_version
```

Every refusal raises `OpenQASMImportError`, a `CompilationError` subclass that
exposes `issue_code` and `line`. This is the first exception in the package to
carry an `issue_code`, so the field is introduced here rather than in a shared
error base that would have no second consumer yet.

The one-register rule is checked where the declaration appears, not where its
absence is felt. A second `qreg` or `creg`, or a redefinition of the first, is
refused as `malformed_statement` naming the offending statement. Left to the
statement parser, the same source instead reported an operand that names an
undeclared register (`unknown_register`) or an index outside a register whose
size a later declaration had changed (`out_of_range_qubit`), which describes
what the redefinition broke rather than the redefinition.

`contracts/openqasm-import-v1-candidate.json` is the machine-readable form of
this decision, and `tools/check_openqasm_import_contract.py` is its reader: it
reconciles the refusal vocabulary with `openqasm_import._ISSUE_CODES` and with
every literal `_refuse(...)` call site, the version lanes and the signature with
the shipped importer, the root additions with `fq.__all__`, and requires each of
the nine declared rules to name a reader that exists — a check in that tool or a
test function in `tests/test_openqasm_import.py`.

### Both directions read one gate table

The emitter, the conformance checker, and the importer previously held three
copies of the same gate spellings. This change moves them into
`flagquantum/compiler/openqasm_gates.py`:

| Table | Entries | Consumers |
| --- | --- | --- |
| `EMITTED_GATES` | 24 opcode to spelling | emitter |
| `FIXED_GATES` | 16 spelling to opcode | importer, conformance checker |
| `PARAMETERIZED_GATES` | 12 spelling to parameter names | importer, conformance checker |

The two inline dictionaries in `target_conformance.py` are now aliases of the
shared tables, and the inline dictionary in `openqasm.py` is deleted; a spelling
that one direction accepts and the other writes can no longer appear.

### Two textual spellings with one meaning

OpenQASM 3 has one three-angle gate `U(theta, phi, lbd)` and no two-angle gate, so
`u2(phi, lbd)` is written as `U(pi/2, phi, lbd)`. Import reads `U` as `u3`
throughout, which makes the two spellings the same program:

| FlagQuantum opcode | OpenQASM 3 text | Imported as |
| --- | --- | --- |
| `u3` | `U(0.4, 0.5, 0.6) q[0];` | `u3` |
| `u2` | `U(1.5707963267948966, 0.1, 0.2) q[0];` | `u3` |

The inverse of `sx` is written in OpenQASM 3 as `pow(-1) @ sx q[0];`. Import
accepts that one power and refuses every other, because the emitter writes no
other inverse and evaluating an arbitrary power would be a second implementation
of a gate the compiler does not own.

## Compatibility

Additive only. `fq.from_openqasm` is a new root name; no existing signature,
default, result field, exception behaviour, or serialized schema changes.
`IR_VERSION` stays `1.0`, because import happens at construction time and
produces an ordinary `CircuitIR`.

`contracts/public-api-v1-candidate.json` records the addition in
`stable_core.retain` and raises `rules.root_export_budget` from 34 to 35.
`contracts/openqasm-import-v1-candidate.json` records the authorization, the
signature, the refusal vocabulary, and the accepted versions.
`docs/public_api_v1.json` and `docs/generated/STABLE_API.md` list the new export.
`tools/public_api_snapshot.py` folds the new contract's `root_additions` into its
authorized set and protects the new signature.
`tools/check_openqasm_import_contract.py` reads the interchange contract itself,
runs in the `quality` job, and is a `tools/pre_push.py` check.

This proposal is not the `FQ-PAULI-QUANTUM-INFO-BOUNDARY-20260930` change and
does not unblock it; that change alters the accepted inputs of an existing Stable
Core name and still needs the API-owner path.

## Acceptance evidence

- `tests/test_openqasm_import.py` (96 collected cases): the export-import-run
  journey in both versions, the sampling path through
  `fq.run(..., outputs=fq.counts())`, the canonical IR including its measurement
  node, all 31 unitary opcodes in both versions with a statevector comparison (62
  cases; the four channels are skipped because they are not gates), comment and
  spacing tolerance, measurement permutation and whole-register mapping, the
  deprecated alias warning, the non-text `TypeError`, and all eleven refusal
  codes.
- `python tools/public_api_snapshot.py` reports
  `public API migration baseline passed`.
- `python tools/check_openqasm_import_contract.py` reports
  `OpenQASM import contract passed: 11 refusal codes, 2 version lanes, 9 rules
  each with a reader`, and `tests/unit/test_openqasm_import_contract.py` drives
  it with one deliberate drift per claim so a field that stopped being read
  fails there. `tests/unit/test_openqasm_import_boundary.py` proves the importer
  pulls in no runtime or simulation module in a fresh interpreter, and that the
  contract gate is a required CI step and a pre-push check.
- `python tools/operator_manifest.py --check` and
  `python tools/docs_source_of_truth.py --check` pass.
- The nine existing emitter tests (`tests/test_openqasm_compiler.py`) and the four
  conformance tests (`tests/hybrid_compiler/test_target_conformance.py`) pass
  unchanged, which is the replacement evidence that the shared gate table did not
  change either direction's behaviour.

## Scope

Import does not evaluate symbolic parameters: `rx(theta) q[0];` is refused as
`unbound_parameter`, so a parameterized program must be bound before export. One
quantum and one classical register are accepted; a program declaring two of
either is refused as `malformed_statement`. `barrier` and `reset` are not
opcodes: their indexed form (`barrier q[0];`, `reset q[0];`) is refused as
`unknown_gate`, and their whole-register form (`barrier q;`) fails one step
earlier as `malformed_statement` because the emitter only ever writes operand
lists of indexed qubits. No emitter output can contain either, so neither
refusal is reachable from canonical text. The measurement block must be terminal,
must define every classical bit exactly once, and each bit must read a distinct
qubit. Import is not a general OpenQASM parser and does not claim to read
arbitrary third-party OpenQASM.
