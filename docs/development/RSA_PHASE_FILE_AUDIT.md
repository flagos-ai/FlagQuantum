# RSA phase estimation and package declarations

The locked reference is Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers
`qualtran/bloqs/cryptography/rsa/__init__.py`, `rsa.ipynb`,
`rsa_phase_estimate.py` and `rsa_phase_estimate_test.py`.

## Source declaration and bound circuit

The source header documents RSA factoring and re-exports its two
operations, ModExp and RSAPhaseEstimate. The native modules expose those
operations as ModularExponentiation and RSAPhaseEstimation through their
algorithm implementation paths. Both are used in the native tutorial.

The unsupplied factoring base and the concrete and symbolic modulus
sizing share a single helper with native modular exponentiation. The
empty source signature and the symbolic call graph are modeled
independently of the executable binding. The source graph declares
one `MeasureQFT(n)` and `2n` modular multipliers. The frozen composite
prepares `2n` PlusState bits, initializes the modular target to 1,
performs its ordered controlled multiplication chain and passes
all `2n` bits to MeasureQFT. It dirty-releases the target. The native
append binding uses the same composition and exposes independent
result slots and the caller's retained modular target.

The tests resolve the source's original input and returned Soquets, the
squared constants and the readout destinations. The native prefix
matches the verified modular exponentiation gate order; the QFT
transform and output measurements follow the original physical wire
order on permuted bindings. Independent DFT sums compare the full
complex state for all 16 terminal results at modulus 3 and all 64 at
modulus 5, including zero-probability outcomes, correlated modular
targets and all clean scratch. Balanced arithmetic erasure histories
are independently covered by the shared modular exponentiation tests.

| Width and modulus | Native primitive counts | Frozen declared model |
|---|---|---|
| 2, 3 | 824 T/T†, 2,096 CX, 600 H, 6 CP, 2 SWAP, 196 measurements, 19 bound wires | 193 AND, 8 CSwap, 1,397 Clifford, 195 measurements |
| 3, 5 | 2,574 T/T†, 6,804 CX, 1,884 H, 15 CP, 3 SWAP, 618 measurements, 25 bound wires | 1 T, 615 AND, 18 CSwap, 4,545 Clifford, 618 measurements |

The declared model has the smaller Fourier child and excludes the
explicit initializer and PlusState preparation. Its QubitCount query
fails at the unused source measurement results. Matching measurement
totals in the second fixture combine different arithmetic and Fourier
counting rules. The executable binding inherits the modulus-fit and
coprime admission conditions. A factory request for modulus 8 keeps
the source exponent width of six and has no admitted lowering with
the three-bit result.

## Tutorial and test disposition

The 22-cell source notebook ran in its source kernel with the original
test file, which reported six passes, four skips and one expected
failure. The [native RSA notebook](../../examples/notebooks/register_rsa_phase_estimation.ipynb)
preserves the three modular exponentiation instances, both phase
instances, symbolic and concrete signatures, and direct call graphs.
It adds a small concrete append example with explicit terminal slots.
The six native code cells and four SVGs ran in a clean native kernel;
the previous worked factoring tutorial also passed again.

Q-0468, Q-0470 and Q-0474 close. Q-0473 remains open for the recorded
source width/model and inherited modulus-fit boundaries, and the
unverified large physical circuits.
