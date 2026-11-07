# Classical basis-vector values

Simulation owns `classical_basis_vectors.py`, adapted from the unchanged
Qualtran v0.7.0 `_ZVector` and `_IntVector` methods in `basic_gates/z_basis.py`
at `bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler retains the accepted
basis/int state emitters and effect bindings. Tensor-network projection remains
with the existing numerical contraction entry, not the raw value callback.

The bit path checks the original width-one restriction before converting bit
truthiness to a literal integer. Creation requires no incoming value; consumption
asserts equality with the configured bit. The integer path checks constructor
metadata first, then applies the original state/effect assertions. Creation
returns the configured object without copying it; consumption returns no output.
There is no extra integer-type guard or width normalization.

The source integer constructor bypasses its bounds only for `sympy.Expr`,
including symbolic integer literals. A Python integer subclass reporting
`is_symbolic()` still receives ordinary bounds checks. This is distinct from
the general symbolic-object protocol and from integer register codecs.
SymPy is optional for native numeric values; a missing SymPy package preserves
the numeric checks, while a nested missing dependency is propagated. No
foreign graph or register type enters the production interface.

Run `python -m examples.algorithms.classical_basis_values` for
`circuit.append(op)`, numerical creation/checked consumption, full complex state
preparation, spectator-preserving effect contraction and inverse cleanup.
Preparing three-bit value five emits two X gates, zero T, zero measurements and
no work. Its source model counts one abstract Clifford component and three
logical wires; the example declares a fourth spectator wire. Effect contraction
is unnormalized and does not imply deterministic reset or memory release.

Reference tests cover constructor-before-callback failure order, unused metadata,
NumPy/symbolic values and exact output identities. Every one-, two- and three-bit
configured state is connected to the existing preparation/effect consumers.
Existing source gate-flow, complex projection and resource suites remain in the
concentrated regression. Whole-file and generic random-measurement protocols
remain outside this accepted slice.
