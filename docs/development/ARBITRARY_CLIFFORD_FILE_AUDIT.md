# Arbitrary Clifford source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0316 | `arbitrary_clifford.py` | `619ac92751b0d6810c5b7087cb0fb1c284edcf44` | Closed for the opaque symbolic and concrete resource leaf. |
| Q-0317 | `arbitrary_clifford_test.py` | `1ccdd85db75eb822087f86ac7dd354ca98ab7192` | Closed by equivalent fresh-process import tests. |

The Compiler's existing arbitrary Clifford key owns the original atomic
resource identity. Its signature is one opaque through register named
`x`. This preserves the zero-width port and the exact unconstrained
symbolic width. The adjoint preserves the signature and uses its own
leaf identity and original label. The peak comes from the signature.
The terminal QEC query reports one Clifford in either direction.
Concrete consumer bindings retain their own bit order. The opaque leaf
owns no numerical execution, work, measurements or binding mutations.

The source supplies no gate decomposition or tensor simulation for this
abstract identity. The native model likewise has no child declarations
or physical emission. The one Clifford is an independent estimate, so
no primitive stream or complex amplitude is inferred from this source
leaf. Integer state/effect and approximate controlled swap use their
existing physical and numerical implementations. Their resource queries
reuse the leaf and preserve the original approximate-swap width classes
and weights in both directions. Native constructors retain the existing
requirement for a nonnegative concrete width.

The two files contain no tutorial. The original test invokes Python in a
fresh process to catch circular imports. The native equivalent imports
the compiler resource owner in a fresh process with an explicit timeout.
Both checks passed without a reference-source change.
