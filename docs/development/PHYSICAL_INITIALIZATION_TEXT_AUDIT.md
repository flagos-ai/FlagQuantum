# Physical simulator initialization text

## Complete source and ownership

This batch closes the complete locked Kickmix `sim/sim.cc` at
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. Its definitions are initialization
record equality, rendering, single parsing and comma-separated parsing.
Compiler owns the unresolved format value and grammar in
`formats/_register_initialization.py`. The value reuses Core
`FixedWidthInteger`, including its declared width and value equality.
Ecosystem retains circuit-register binding and command admission.
Simulation retains numerical initialization and execution.

The internal format value is needed because the existing numerical
`RegisterInitialization` stores resolved members. Parsing an initialization
string needs an unresolved uint32 register index and the literal's declared
width. No new stable namespace export or circuit entry is introduced.

## Interface and semantics

`parse_initialization` returns one immutable unresolved record.
`parse_initializations` returns an immutable tuple. Register bits remain
LSB-first when bound to existing physical register members. `str(record)`
normalizes wildcard input to `<0` exactly as the source does. Absent register
index digits select zero. A single trailing comma is accepted, an interior or
additional empty item is rejected. Literal spaces follow the Core parser.
The uint32 index bound is enforced before Python integer conversion can reach
the interpreter's decimal-digit limit. Leading zeros retain source acceptance.

A parsed zero-width random zero is distinct from a nonzero-width zero literal.
Existing command binding retains its zero-modulus rejection for the latter.
Unresolved valid uint32 indices can be parsed independently from a circuit, while
binding rejects undeclared registers. Ordered repeated assignments retain their
existing overwrite behavior.

## Definition disposition

| Complete source definition | Native owner |
| --- | --- |
| SimInitInstruction equality | Immutable record compares index, Core value and random flag. Core equality retains numeric equality across declared widths. |
| operator<< | Format record renders index, assignment/random operator and existing decimal value. |
| from_str | Compiler grammar retains absent digits, uint32 bounds, wildcard, both operators and existing Core integer factories. |
| from_str_many | Compiler splits ordered records, retains one trailing separator and appends the source examples suffix on errors. |

## Source and native error boundary

Grammar/index errors and examples suffix retain the original strings.
Malformed scalar payloads retain the authoritative existing Core error vocabulary.
The original reports the invalid character and byte value or an empty-int error.
Core reports invalid unsigned digits or empty integer text. Differential tests
record both outcomes separately and require matching rejection. Source messages
are retained verbatim in the fixture. They are not rewritten into the native
messages or presented as byte-identical error evidence.

## Verification and resources

The complete four-file simulator intake verifies 92 source/dependency blobs and
runs all 15 unchanged simulator tests plus two direct helper tests successfully.
The initialization oracle separately verifies 86 frozen blobs and captures 62
single/multiple parsing cases. It preserves each original record's index,
declared width, hexadecimal storage, random flag, normalized rendering and
ordered equality pairs. Native format/command checks pass, including parser
replacement through the existing Ecosystem binding and actual numerical entry.
The final batch passes 1,180 concentrated checks, 121 integration checks and
strict typing for four Python files. The new parser has 100% statement and branch
coverage. Regeneration is byte-exact. Black, Ruff, scope, architecture, IR and API
checks pass. The existing 720 dependency findings are unchanged as a complete
line set. Locked Kickmix 0.1, Simulator and extension SHA-256 are verified.
Final receipts are recorded externally as `ftqc-reference-env/g2-initialization-*-20261123.*`.

Parsing/binding emits zero primitives, measurements and quantum workspace.
The actual four-lane initialized X/CX consumer emits four X and four CX, with
exact zero phase and zero final quantum outputs. Analytical estimation is
inapplicable and recorded separately as `None`. Existing complete complex-state,
measurement-branch and workspace checks are rerun at batch acceptance.

## Remaining simulator group

The full `sim.h`, `sim.test.cc` and `sim.perf.cc` remain open. Persistent condition
stacks across scans, full debug callback masks including inactive lanes, state
capacity/buffer lifecycle, numerical replacement and benchmark disposition still
need complete source evidence. No complete simulator-header or SIMD runtime
closure is claimed by accepting the separate initialization source file.
