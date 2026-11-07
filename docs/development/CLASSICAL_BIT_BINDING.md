# Classical bit binding through shared codecs

Simulation owns `classical_bit_binding.py`, ported from the unchanged Qualtran
v0.7.0 Allocate, Free, Split, Join and Cast numerical callbacks under
`bloqs/bookkeeping` at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The native boundary accepts existing scalar codec callables, not foreign dtype
objects. It creates no parallel register hierarchy or numerical executor.

| Callback | Original operation order |
| --- | --- |
| Allocate | Build `[0] * width`, then decode; dirty metadata does not alter this callback |
| Free | Compare value with zero, then check dirty permission, then return no values |
| Split | Encode, then `np.asarray`, without forcing a bit dtype |
| Join | Call the input array's `tolist`, then decode |
| Cast | Pass the exact encoder output object directly to the destination decoder |

Input/output codecs own their numerical domains, scalar types and failure
behavior. Compiler retains register-layout legality, static width checks and
workspace leases. The callbacks consume bound metadata and do not replace the
original dtype or Cast constructors. Compiler's stricter integer constant
reinterpretation remains a compile-time path, not the raw numerical callback.
No production dependency points from Simulation into Compiler.

Tests use native bit, signed/unsigned integer, fixed-point and Montgomery codecs
against the unchanged source callbacks. They retain NumPy dtype inference,
malformed-input failures and output types. Replacement probes verify that codec
calls are actually consumed, Cast preserves the encoded object's identity, and
Join performs the required list conversion. Free still fails on an ambiguous
array comparison even with dirty permission; it does not bypass the comparison.

Run `python -m examples.algorithms.classical_bit_binding` for the ordinary
`circuit.append(op)` path. It compares reversed signed register views with raw
Split/Join/Cast evaluation across complete complex inputs and a spectator,
then restores the input by the inverse. The metadata operations emit no gates;
the explicit bit action emits one X. There are no T gates, measurements or work.
Existing source binding/resource and workspace measured-branch tests remain in
the concentrated regression. Numerical Free is neither a reset nor a guarantee
that quantum storage has been returned to zero or released.
