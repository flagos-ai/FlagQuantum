# Register data-type tutorial audit

The complete 24-cell `qualtran/DataTypes.ipynb` at revision
`bcb17bb25360ed68ce3bc084645db3e929799389` has Git blob
`edba7005d69b811369ec869e13b3ce77ea33880c`. The original executes unchanged.
`examples/notebooks/register_data_types.ipynb` preserves every cell's kind and
records its original index in `frozen_source_cell`. All prose is adapted to the
native owners and all code uses existing FlagQuantum paths.

| Source cells | Native responsibility and evidence |
|---|---|
| 0–2 | Compiler builder allocation, one H call and explicit output ownership transfer prepare zero and plus. Core owns linear handles. |
| 3–6 | Existing Signature and Register constructors report two qubits and two 32-bit unsigned words. |
| 7–8 | A 2048-bit unsigned signature and two existing ArithmeticCall identities describe BitwiseNot then AddK(1). The declared graph retains both leaf multiplicities. |
| 9–10 | Existing split, shared bit-gate call and join construct the complete three-bit NOT Program. Bind and append use the existing circuit entry. |
| 11–12 | Simulation integer codec gives the same eight MSB-first bits for 0x30. |
| 13–14 | Existing allocation, default opaque join and typed cast return one unsigned zero word. Physical output leases explicitly transfer to the caller. |
| 15–16 | Existing loose and strict compatibility predicates reproduce all seven source results. |
| 17–20 | Nineteen native declarations retain exact source total, quantum and classical widths. Metadata storage fields replace foreign base-class checks and display names. |
| 21–22 | Existing domain and scalar-codec owners reproduce first/last encodings for all eighteen numerical declarations. Opaque domain remains ambiguous. Field and polynomial numerical objects stay with Simulation. |
| 23 | Existing callback vectorizers and domain ownership explain extending numerical encoding. Protected declaration and Program changes follow the API proposal process. |

## Execution and resources

The actual NOT graph has Split, three X calls and Join in the same source order.
The bound stream has X on physical wires 0, 1 and 2. Full complex amplitudes agree
with the unmodified source tensor on an entangled spectator state. Applying the
same operation twice restores the complete state. The source QEC model reports
three Clifford gates, zero T and zero measurements. No work or measurement is
allocated by this operation. The preparation emits one H; the cast emits no
primitive. Their output allocations remain live and explicitly transfer to the
caller. Their physical order follows the existing workspace allocator.

The large negation example constructs a declaration graph. Its two children
and declared width are compared with the original source. It does not execute
a 2048-qubit statevector. Numerical type tables likewise authorize no wider
classical Program admission. That separate contract remains Proposal 108.

The native notebook executes while imports of Qualtran, Cirq and Kickmix are
blocked. Reference-only tests execute the unchanged original and compare its
three complete program examples, width properties and domain endpoints.
Full data-type implementation evidence remains in DATA_TYPE_SOURCE_FILE_AUDIT.md.

## Acceptance

All four complete notebook scenarios pass. The concentrated data-type regression
passes 6,388 cases with 205 existing numerical warnings. Strict typing checks the
test module and every flattened native code cell. Black, Ruff, team scope,
architecture and IR checks pass. Dependency inspection reports 688 existing
repository findings and zero changed-path findings. Source and native notebooks
both execute all their code cells. The native output uses encoding labels and
metadata fields in place of foreign class names and inheritance predicates.

The default suite retains the pre-existing public-docstring coverage failure.
Its matching baseline and the exact results are recorded under
`g2-datatype-tutorial-default*`. No all-repository passing claim is made.

Evidence is recorded under `ftqc-reference-env/g2-datatype-tutorial-*`.
The tutorial adds no production API or encoding implementation. The complete
native owner and dependency inventory is inherited from the accepted Q-0160
and Q-0161 audits. Q-0141 is closed by this 24-cell replacement.
