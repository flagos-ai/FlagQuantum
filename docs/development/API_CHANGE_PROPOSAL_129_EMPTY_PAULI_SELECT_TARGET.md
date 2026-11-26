# Proposal 129: empty-target Pauli SELECT

Status: explicitly approved by the user on 2026-11-26; implemented and locally
verified as documented below.

## Verified source requirement

Unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, executes SelectPauliLCU with
target_bitsize=0. Empty Pauli words carry only a sign. For a one-bit selection
and the two words empty/negative-empty, the complete source operator is
diag(1, -1) on selection. An optional zero/one control makes that phase
conditional; ignoring it would lose observable relative phase.

Nine actual combinations cover selection widths 1/2, tables of 2/3/4 rows and
absent/zero/one controls. All original root graphs, complex tensors, source QEC
models and logical peak queries succeed. The source implementation blob remains
`41e2f8ddb341fbbdd2d004ef533041bd24a5898c`. Evidence, including full operator
diagonals and current native refusals, is in
ftqc-reference-env/g2-empty-pauli-target-intake-20261126.json beside the worktree.

Current native append raises IRValidationError, invalid select_pauli_lcu
operands, because Core requires a positive target width. Stored native Program
capture separately refuses in the Pauli component owner. The source examples
use actual selection/control qubits; zero-qubit physical circuits are not needed.

## Proposed contract

- Admit target_width=0 for the existing structured select_pauli_lcu operation.
  Retain strict Python integer metadata, nonnegative widths, wire cardinality,
  distinct operands and all existing control/table validation.
- Each zero-target word is either the empty string or a single minus sign.
  Populated addresses apply their signed scalar to the selected sector. The
  existing logical path retains identity on unused addresses; bounded source
  traversal retains its separately verified padded-address behavior.
- Keep selected_majorana_fermion's positive-target requirement. Negative target
  widths and inconsistent words remain invalid. Existing empty-selection
  source traversal refusal and zero-wire circuit rules remain unchanged.
- Retain circuit.append(op), current constructor/binding methods, codecs and
  instruction fields. Add no public export, opcode or serialization vocabulary;
  this extends one existing parameter's accepted value domain.

## Implementation ownership

Core owns the operation-specific admission condition. Compiler's existing Pauli
component, bounded traversal and dense/unitary lowering own signed phase calls,
empty target handles and source-ordered predicates. Algorithms keeps its current
operation entry; Ecosystem converts the same external metadata. Simulation
continues to execute existing primitives/matrices. No alternate runtime, target
qubit substitute, allocator or source callback is introduced.

## Acceptance and compatibility

Compare the nine actual source families in both directions, ordered calls,
primitive dependencies/operands/feedback, complete complex operators, all small
measurement branches, private work zero and restored empty target ownership.
Verify normal append and JSON/Program round trips. Record actual primitive
inventories separately from independent source QEC models and peaks.

Verify all previously accepted positive-target Programs and finite metadata
streams remain identical. Check invalid negative widths, malformed sign words,
nonbinary controls, duplicate/wrong-count wires, empty physical circuits and
Majorana zero-target refusal. Preserve all original source failures and the
distinction between SDK, stored graph and resource protocols. Run affected
regression, local integration, strict typing, format, architecture and API checks.

The [Core boundary](../../flagquantum/core/AGENTS.md) requires an integration
contract before protected IR changes. AGENTS.md rule 8 and the
[API protection policy](PUBLIC_API_PROTECTION.md) require explicit user approval
before implementing this accepted-value extension. Earlier scalar and empty
storage proposals do not authorize this structured operation change.

## Implemented evidence

Core now permits the empty Pauli target while retaining positive Majorana
targets and nonempty physical operands. Compiler retains its positive-width
wire composer and carries the empty logical target through a typed wrapper,
then inlines that wrapper with the existing call expander. It assigns no
physical wire or substitute qubit to the empty word.

Fifteen source table/control families, including constant positive/negative
tables, pass both directional call comparisons and every logical operator
column in every small measurement branch. Source peaks and independent QEC
categories match thirty recorded directional primitive/model inventories.
All three logical lowering paths and Boolean empty-target metadata pass.
The original empty-selection source refusal is retained independently of the
existing native logical controlled-constant path.

Logical and primitive IR use their real JSON codecs. Internal ProgramRecord
object round trips use Python pickle only as a test identity check; no new
public Program serialization profile is introduced. The existing native
physical-context codec continues refusing contextless logical Programs.

The batch passes 143 affected cases in each Cirq 1.7/1.6 profile, 145 local
integration cases and 79 existing Core IR/operand cases. Nine Python files
pass strict typing, format and lint; ownership, language/links, architecture,
IR and API checks pass. Twelve old operations and twenty-four complete
positive-target Programs remain equal to the published baseline. The shared
test fixture and source compiler retain identical AST bodies after extraction.
