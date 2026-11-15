# Double-factorization file audit

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/df`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0340 | `double_factorization.ipynb` | `1b46c50ef421c01ff48008bc7479787b0ab14b05` |
| Q-0341 | `double_factorization.py` | `e83b385240a5e98b8ab939f6e7eef0fff1547e5c` |
| Q-0342 | `double_factorization_test.py` | `a5a48d8cf2444c21349d61daeed6a4da4d5e2653` |

## Owners and executable boundary

Algorithms owns the one-body and full block resource declarations and the
coefficient-driven block. PREPARE, indexed metadata, programmed rotations and
number phase remain separate component owners. The block uses
`circuit.append(op)`. Persistent signal registers retain their declared
ownership across the block and explicit adjoint. The transient scratch belongs
to the compiler and is cleared after use.

The native tutorial at `examples/notebooks/register_double_factorization.ipynb`
preserves the two original examples with 10 spin orbitals and 250 eigenvalues.
It displays their direction, word widths, array shapes and depth-one graphs,
including leaf multiplicities. The full block score matches the seven ordered
calls and their operand sets in the original decomposition. The middle
reflection uses the outer success bit, the orbital index and spin. A smaller
example constructs and appends an executable block.

Run the tutorial regression with
`python -m pytest tests/integration/test_register_graph_notebook.py -k register_double_factorization`.

## Evidence and resource accounting

- The original standalone suite ran with OpenFermion 1.6.1 and reported 11
  passed and one skipped, with notebook execution selected separately. It
  includes the 108-spin, 360-factor and 13,031-eigenvalue comparison. The native
  model checks the complete ordered source graph at that size.
- The original full notebook model declares 1,245 Toffoli, 65 measurements
  and a 95-qubit peak. The estimate runs with emission disabled.
- The independent coefficient fixture from the prior executable batch emits
  482 T/T† and 89 measurements with four work qubits and 42 reserved wires.
  Its full complex projection is checked against a two-spin fermionic
  reference, including four representative measurement histories and work
  cleanup. Prior tests also cover forward/adjoint recovery on coherent
  nonzero signals. These measurements refer to the executable fixture and
  are reported separately from the large model.

## Remaining source boundary

The source coefficient factory and the one-body and block `alpha` and
`epsilon` properties raise `NotImplementedError`. The supplied cost-only
nodes only declare resources. The source also omits the `l_ne_zero`
predicate in the one-body number-phase and middle-reflection paths
(Qualtran #1022). The native coefficient path implements the intended
controlled behavior; its projection and primitive execution have independent
tests. Q-0341 remains open for those source gaps, arbitrary external
subcomponents and literal full-block primitive parity.
