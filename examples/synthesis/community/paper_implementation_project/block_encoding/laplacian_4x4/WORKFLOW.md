# Two-dimensional Laplacian block encoding

The native program uses one complete three-bit selector array, two two-bit
coordinates, one data bit and one XOR boundary flag. The output order
`s,j1,j0,data,dlt` exactly preserves the source's packed `s,j,data,dlt` order.
Array index zero is the low selector bit; `j1` is the high coordinate.

`inplace_prepare_state` applies the existing real preparation unitary to all
eight selector columns. `within`/`apply` reverses exactly that unitary after the
block, even though the block changes the selector. Complete-register Fourier
shifts with explicit carry and borrow conditions reproduce the source's full
position increments and coordinate increments. No subregister QFT API is added.
The boundary flag is XORed. Centered conditional phase between data Hadamards
reproduces the three RX entries, including the absolute minus sign of RX(2pi).

The independent Circuit directly enumerates the source's 512 basis columns,
modular shifts, flag toggles, Z and RX matrices. Tests compare the untouched
original, native program and reference full complex state, plus two nonzero
angle/probability bindings covering all selector branches and all four initial
data/flag words. They also check bit order, absolute phase, zero work, reference
unitarity and resource refusal. The complete Notebook workflow remains separate.

Current default native costs are 9 physical bits, 0 workspace, 896 operations and
depth 725. The original uses 12 physical bits, 139 operations and depth 113.
The native example saves three qubits but uses more operations and depth.
Efficient controlled modular arithmetic remains a concrete optimizer gap.
