# Complete Kikuchi list and matrix notebooks

Q-0601 and Q-0604 map the complete frozen Qualtran v0.7.0 notebooks
`kikuchi_adjacency_list.ipynb` and `kikuchi_adjacency_matrix.ipynb` to the native
notebooks with matching basenames under `examples/notebooks/`. Their frozen
blobs are `7ed33e384065876e374a0eb63c88676f09c01b0e` and
`599e88a21a42bfe9ca32ce7bc08277f7b5aa44f1` at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

All ten list cells and eleven matrix cells retain their source-cell identity.
Each notebook adds one explicitly native model-evidence cell. Both original
notebooks execute completely and unchanged. Native cells run with foreign SDK
imports blocked. The examples use Algorithms-owned instances and declarations,
Core-owned signatures and Drawer-owned graph rendering. The notebooks retain
source mathematical intent and state the current physical acceptance boundaries.

The list example reproduces ell=8, brute-force sparsity, array signatures and the
two distinct forward/reverse graph children. The matrix example reproduces both
symbolic ell=c*k and the concrete ell=8 example with three-bit angle metadata,
packed signatures and five compute/rotation/uncompute children. All source
signature shapes, types, sides, instance scopes, graph roles/multiplicities and
seven concrete/symbolic QEC categories agree. The five rendered signature/call
graphs are visually inspected through a contact sheet in the evidence directory.

Source QEC models remain separate from emitted primitives. These notebook
examples do not establish root physical execution, complex amplitudes,
measurement histories or zero restoration. Q-0602/0603 retain missing query
bodies and the literal root return/free discrepancy. Q-0605/0606 retain the
matrix physical/control-adder and original source-test limitations. Their
existing source and executable child evidence is reused from
`KIKUCHI_ADJACENCY_FILE_AUDIT.md`, `KIKUCHI_MATRIX_FILE_AUDIT.md` and
`KXOR_LOADING_FILE_AUDIT.md`. Existing executable children retain the
`circuit.append(op)` workflow.

The complete notebooks and their direct existing dependencies participate in
batch regression, strict checking of both extracted scripts and their native
test, formatting, lint, scope, architecture, IR and protected API checks.

Concentrated regression passed 562 tests with four visible warnings. Black,
Ruff, strict mypy across both extracted scripts and the test, Algorithms scope,
architecture, multi-level IR and protected API checks passed.
