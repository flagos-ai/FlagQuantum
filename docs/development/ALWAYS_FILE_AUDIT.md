# Always source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0313 | `always.ipynb` | `5e127210038748d4b79ad8cb012b63f263139363` | Closed by the full native tutorial. |
| Q-0314 | `always.py` | `905f79a2fef8cf14ffd3f09098d4d93ea356a250` | Open for arbitrary external objects and generic typed child resource delegation, including source atom diagnostics. |
| Q-0315 | `always_test.py` | `2b712e48f42fdf79519c7a5493423815fe52872f` | Closed for the runnable original cases and notebook. |

The Compiler wrapper passes the inherited control tuple back unchanged and
executes the child emitter with its own bound operands. The optional typed
signature preserves source port order and sides. The adjoint swaps the
forward and inverse emitters and reverses the sides of one-way ports.
For the original And example the control register is an ordered pair of
scalar through bits, and the clean target is output-only. The target is
input-only for erase. The caller owns that physical wire and the result
slot. The inherited control neither changes the child's signature nor
appears in the child stream.

The fixture checks every operand, measurement slot, feedback condition
and reset in the unchanged source stream. An arbitrary complex input
spans the inherited control and both And controls. The compute map agrees
on every component. Both unnormalized erase histories restore all
amplitudes and clear the target bit. The And forward stream emits four
T/T-dagger instructions. Erase emits H, a measurement, a feed-forward CZ,
and reset. The child logical width is three and the demonstration reserves
four wires including the inherited control.

The source displays one atomic Always(And) leaf, in either direction.
Its aggregate QEC query raises `DecomposeTypeError` through the atomic
child. The independent native And model matches the source And child
query. This covers child cost, and the wrapper graph records the source
leaf and diagnostic separately. The existing Always(Add) model copies
direct child calls without adding an artificial edge and matches the
original Add counts and peak. Existing controlled AddK cases verify both
control polarities, both directions and every small measurement history.

The unchanged original non-notebook run has five passes and one expected
serialization failure. The unchanged ten-cell original and twelve-cell
native tutorial pass independent kernels. The native tutorial preserves
the original usage patterns and the And signature and leaf graph. It then
appends the compute and measured erase to a coherent state.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_always.ipynb
```
