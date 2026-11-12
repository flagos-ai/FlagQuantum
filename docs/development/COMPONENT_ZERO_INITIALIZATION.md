# Zero-input component initialization (GRAPH61)

`compiler/registers/component_zero_initialization.py` adapts the unchanged
Qualtran v0.7.0 `_add_classical_kets` and `initialize_from_zero` call chain,
pinned at bcb17bb25360ed68ce3bc084645db3e929799389.

`initialize_component_from_zero(record, entry="main", name="from_zero")`
adds a typed no-input wrapper. Every left register element is prepared in
signature and row-major element order, followed by one call to the original
entry. Outputs retain their names, dtypes, shapes and order with RIGHT direction.
Original FunctionRecord objects and record metadata are retained; the derived
canonical payload is invalidated. This owns Compiler graph construction only.

Preparation uses existing `allocate_word_function(..., dirty=False)` definitions,
shared by dtype within the wrapper. Their contract supplies clean owned zero
words without reset gates. Generated names avoid existing definition prefixes.
Physical binding requires caller-supplied clean wires. Returned prepared words
transfer live ownership only when `allow_workspace_outputs=True`; the wrapper
does not certify those output data bits are zero after computation.

No new IR, graph type, preparation registry, numerical kernel, resource model or
public API is added. Classical inputs, invalid modules, untyped/physical entries,
unknown entries and wrapper-name collisions are rejected. Empty arrays add no
preparation calls; zero-width words retain logical ownership.

Twelve width/shape layouts compare the original preparation call widths, count
and order, output declarations and ownership. Six positive-width nonempty
layouts additionally compare the unchanged source gate stream (names, operands,
parameters and per-wire dependencies) against the bound native initialized graph.
All emit three H and two S gates, preserving complex phases. The wrapper adds
no primitive gate. Namespace collisions, record metadata, empty-input wrappers,
invalid/classical entry rejection and explicit output ownership are covered.

The original source cannot finalize empty inferred output arrays (IndexError).
Its zero-width IntState decomposition fails with a QAny(1)/QAny(0) dtype error.
These original failures remain negative evidence. Native declared-signature
behavior is verified independently and is not labeled source contraction parity.
No reference source is patched.

Simulation separately supplies zero kets through `register_zero_state_nodes`;
see [zero-state nodes](REGISTER_ZERO_STATE.md). Compiler never imports that
owner. The caller expands/selects the graph through existing owners and supplies
numerical leaves to the GRAPH60 SSA wiring. Full primitive suppliers,
superoperator/discard semantics and full parent-file acceptance remain open.

The existing user workflow remains `circuit.append(op)`. Its runnable supported
example is `python -m examples.algorithms.binary_field_arithmetic`, which reports
GF(8) multiplication, clean inverse work and sixteen erasure trajectories.
For graph-wrapper changes run:

```bash
python -m pytest tests/unit/test_component_zero_initialization.py -q
```

## Final verification

Compiler implementation `c55f7b12` and separate Simulation implementation
`05c6455b` passed 916 related tests (18.25 seconds), 221 focused tests
(29.02 seconds) and 64 integration tests (8.33 seconds). The new Compiler owner
has 44 statements and 20 branches, all covered; the new Simulation owner has
13 statements and four branches, all covered. Eight selected owners total
263 statements and 112 branches at 99% combined coverage; GRAPH60 retains its
two uncovered profile-refusal branches.

Strict typing, Black and Ruff passed for sixteen files. Both team-scope checks,
architecture, Program contract and protected API checks passed. Twelve original
source blobs match the inventory. Seven unchanged source quimb and basis-vector
tests passed in 1.57 seconds, retaining two original notebook-marker warnings.
The locked Qualtran 0.7.0 and Kickmix 0.1 Simulator were checked, and the existing
binary-field example passed.

The default smoke/unit run reports 21,507 passed, 300 skipped and seven failed
in 227.02 seconds. All seven failure IDs exactly match the independent existing
seven-failure baseline; no new failure ID appeared. Maxfail stops the run early,
so the default suite is neither complete nor green. Related regression retains
five existing warnings, including three from the original Add tensor path.

Resource evidence distinguishes empty physical preparation streams from original
IntState coarse models: three prepared elements change the phase example's
Clifford model from five to eight and the measured Add model from 22 to 25.
Actual phase primitives remain three H/two S, and Add retains sixteen T/Tdag,
four measurements, two clean work bits and original peak eight. Source zero-bit
peak failures and empty-output initialization failures are recorded explicitly.
Logs, source hashes, resource categories and exact baseline comparison are under
`ftqc-reference-env/g2-zero-initialization-*` in the local reference workspace.

## Classical output follow-up

RIGHT-only classical results are now retained: the initializer only allocates
left quantum words and does not prepare output ports. Classical inputs remain
rejected. See `QUIMB_SOURCE_DISPOSITION.md` for initialized H/S followed by X/Z
measurement with retained or discarded results, full density and phase-gradient
comparison, all physical branches and separate actual/model resources.

The physical pool-owned destructive measurement still requires a retirement
ownership contract. Tests retain that refusal and bind the original graph with
caller-owned zero inputs for physical branch and append checks. Numerical
initialization support must not be read as that physical ownership capability.
