# Generic multi-control constructor and target ownership

Reference: unchanged Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`mcmt/multi_control_pauli.py` and direct ControlledViaAnd dependencies.
Existing complete execution, tensor, diagram, model and source-test evidence is
retained in [the file audit](MULTI_PAULI_FILE_AUDIT.md). This batch owns the
missing serialized generic constructor connection, not another gate generator.
Parent Q-0556 is already accepted; Q-0876 remains a larger catalogue scope.

## Interface fixed before implementation

Compiler formats owns the fields `cvs` and resolved `target_bloq`, literal
binary conversion, source register names and per-compilation constructor
recipes. The target must expose one physically compatible scalar quantum-bit
THRU register. The shared call type checker retains dtype admission. Values
are row-major, MSB-first, and stay in the original tuple order.

Zero controls have only the caller-owned `target` port and call the original
child directly. Nonzero controls expose a shaped `controls` QBit array and
`target`. The root calls the existing ControlledViaAnd owner with that exact
array clause, even for a singleton or two positive controls. It must not
replace the latter with a different Toffoli specialization. The predicate and
positive target provider own private work, classical slots and clean erasure.

The format factory accepts an existing positive-control provider. Its default
recognizes the authoritative fixed X/Y/Z/H and directional S/T leaves; other targets require their
explicit native owner and helper closure. A scoped generic adjoint is enabled
only when an outer adjoint fallback is supplied, preserving existing catalogue
merge behavior. Unknown wrapper roots delegate to that same fallback. A target
inverse owner returns a complete native Program. Default exact inversion is
restricted to the four known self-adjoint leaves and the exact S/T inverse pairs. No inverse is inferred from
a signature, name or reversal of measured primitive instructions.

Generic inverse construction retains the source control predicate and replaces
only the child by its owned exact inverse. Recipes are keyed by complete native
functions in the current factory instance. The existing register builder,
namespace handling and enclosing resolver retain external child helpers.
No source framework object, backend selection or numerical kernel enters
Compiler. Users continue to append the resulting native operation normally.

Symbolic HasLength remains an original codec refusal. Invalid target widths,
sides, shapes, metadata and unowned control/inverse bodies fail explicitly.
Actual primitive inventories and directional model estimates are recorded
separately. Registration alone does not close the constructor catalogue.

## Verified delivery

The concentrated closure passes 629 cases in each actual Cirq 1.7.0/Qiskit
2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 profile, with no skips. Qualtran 0.7.0 and
the real Kickmix 0.1 Simulator are asserted. Another 145 local integration
cases pass. Three production/test files pass strict typing, Black and Ruff;
team scope, English/link checks, architecture, internal IR and protected API
checks pass. The exact source multi-control implementation, its tests and
ControlledViaAnd match their frozen Git blobs. Thirty-six existing X/Z complete
Programs and bound IR payloads remain identical to `dcd7a8cf3`.

Six fixed targets, six control patterns and explicit adjoint depths 0/1/2/3
are resolved at source serialization depths 0/1/2. The comparison retains
per-wire gate order, operands, exact angle dictionaries and measurement
feedback. Every complex basis column is checked across every applicable small
measurement history, including a coherent borrowed spectator and zero clean
work output. Resolution runs with Qualtran blocked; normal append and IR JSON
round trips run with source SDKs blocked.

A custom single-port target named `ctrl` exercises collision-safe controls,
complete helper ownership and a child-private measured CZPow workspace. Its
explicit inverse keeps the outer predicate and reverses only logical calls
using an owned inverse leaf. Both directions compare full operators and all
certified erasure histories. This fixture's positive provider explicitly
retains the source SDK T/T-dagger leaf inside the measured CZPow view; no
source file is patched. Unowned controls and inverse bodies, invalid scalar
ports and metadata fail; independent catalogue instances retain their own
fallback dispatch.

## Physical primitives and independent estimates

Seventy-two physical inventory rows are retained separately from the model
checks. All seven available QEC categories match the unchanged source model;
primitive counts are not substituted for that model. For fixed targets with
`n > 0`, the predicate emits `4*(n-1)` T/T-dagger gates and `n-1` measurements,
resets and clean work bits. The positive target remains its own atom: CX, CY,
CZ, CH or directional controlled phase. A named Controlled(ZPow) measured
recipe is not substituted for Controlled(SGate/TGate).

| Target examples | Values | Actual primitive instructions | Actual measurements | Actual T/T-dagger | Independent model |
| --- | --- | ---: | ---: | ---: | --- |
| X, H, S, T | 1 | 1 | 0 | 0 | Positive child model only |
| X, H, S, T | 1, 1 | 18 | 1 | 4 | One And plus directional positive child and cleanup |
| X, H, S, T | 0, 1, 0 | 43 | 2 | 8 | Two And plus signed predicate Clifford calls and child |

This connects a generic shallow constructor to existing native execution;
it does not close whole-library reflection, arbitrary target inverse providers,
symbolic HasLength serialization or every unmatched constructor. The existing
accepted multi-control parent is not counted again. The reconciled catalogue
contains 198 Compiler registrations, 39 Ecosystem registrations and a union of
199/303 keys. The 104 unmatched keys comprise two special graph/descriptors,
five abstract interfaces, one original module-valued key and 96 concrete
classes; absence of a factory does not imply absence of a native algorithm.
