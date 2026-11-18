# Complete Kikuchi Hamiltonian notebook

Q-0607 maps all eleven source cells of the frozen Qualtran v0.7.0
`optimization/k_xor_sat/kikuchi_block_encoding.ipynb` to the native notebook
with the matching basename under `examples/notebooks/`. The source blob is
`78031452acc39de315339e2962f85067473c49a9` at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Copyright and Apache-2.0 provenance
remain in the artifact. One native evidence cell records concrete, symbolic and
positive-control QEC models separately from emitted primitives.

Both the complete unchanged original notebook and native notebook execute.
The native execution blocks Qualtran, Cirq and OpenFermion imports. Examples
reproduce the concrete instance, ell=8, symbolic ell=c*k, default precision,
sparsity, packed signatures and one-child sparse-encoding graph. All seven
source QEC categories agree for both examples. Three signature/call graphs
are rendered and visually inspected through the evidence contact sheet.

The native examples use the Algorithms KikuchiHamiltonian owner, Core
signatures and Drawer rendering. Reused test execution also revalidates the
complete list/matrix notebooks. The physical query, packed-array and source
root return/free limitations stay with Q-0602/0603/0605/0606/0608/0609.
`KIKUCHI_BLOCK_ENCODING_FILE_AUDIT.md` records those boundaries. Notebook
closure does not establish physical root amplitudes, measurement histories,
work-zero restoration or emitted primitive counts. Existing executable shared
components retain circuit.append(op).

The complete notebook test, extracted script and dependency regression receive
strict typing, formatting, lint, scope, architecture, IR and protected API
checks. Final results accompany the synchronized batch ledger.

Final concentrated regression passed 849 tests with three warnings. Strict
mypy passed both the notebook test and extracted native script. Black, Ruff,
Algorithms scope, architecture, multi-level IR and protected API checks passed.
