# Arcsine argument reduction

Appendix D reduces the nonnegative magnitude to the input of the polynomial
in the arcsine approximation. For x < 1/2 that argument is x². At and above
1/2 it is (1 − x)/2. The upper branch later uses its square root and a
π/2 correction. [Appendix D, reversible implementation](https://arxiv.org/pdf/1805.12445#page=10)

The compiler entry accepts a big-endian unsigned source with one integer
bit and f ≥ 1 fractional bits, valid on [0, 1] including both endpoints.
It preserves the source and XORs the selected argument into a big-endian
2f-bit pure fraction. That width represents both branches exactly.
Scratch and classical slots are caller-owned. Quantum scratch begins
and ends at zero. The result can start in any state.

One negative-control comparison records the low branch in a clean flag.
The shifted input is subtracted from a register initialized to 1/2.
Conditional copy, addition and a final X erase that register.
The square of the fractional suffix is computed, copied under the opposite
branch condition, and erased. Its omission of the integer bit is valid
because that branch implies a zero integer bit. The two arithmetic paths
reuse the same scratch wires and slots. The comparison is then erased.

The coherent tests include the 1/2 boundary, x=1, nonzero output,
all four primitive measurement histories at f=1, and the reversible
channel certificate for every legal input through f=8. A dynamic
trajectory test uses `circuit.append(op)` twice to recover the original
state.

| Source fractional bits | Native T/T† | CX | CCX | Measurements | Work wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 20 | 6 | 2 | 4 |
| 2 | 32 | 76 | 9 | 8 | 7 |
| 3 | 72 | 176 | 12 | 18 | 13 |
| 5 | 200 | 508 | 18 | 50 | 31 |
| 8 | 512 | 1,336 | 27 | 128 | 73 |

The counts include cleanup and keep CCX separate. Each measurement
has one reset. Classical slot count equals the source fractional width.

Qualtran v0.7.0's complete ArcSin has a cost formula and a classical
method. It has no gate decomposition for the complete operation or this
stage. At four total bits, the model estimates 2,826, 2,520 and 1,908
Toffoli gates for two, three and four declared fractional bits,
respectively, with default degree and iteration count. Its classical
method instead scales by the total width in all three cases. For input
word 15 it returns 19, beyond the declared four-bit output.
The native table measures only argument reduction, with no polynomial
evaluation, inverse square root, final angle or sign handling.
[Locked ArcSin source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arcsin.py)
