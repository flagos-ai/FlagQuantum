# Always source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/bookkeeping/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0313 | `always.ipynb` | `5e127210038748d4b79ad8cb012b63f263139363` | Closed by the full native tutorial. |
| Q-0314 | `always.py` | `905f79a2fef8cf14ffd3f09098d4d93ea356a250` | Closed by generic native declaration delegation, actual foreign consumers and the accepted concrete source captures. |
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


A native typed wrapper retains its child's complete signature and one
stored call. The common control callback returns the original control
tuple and adds the wrapper through the existing builder. Forward and
adjoint Add fixtures retain their source streams, operands and feedback.
Each direction emits eight T/T-dagger instructions and two measurements.
All four histories per direction agree on full complex amplitudes and
return both clean work bits to zero. The independent Add model and the
source Always query agree. The compiled atomic Allocate fixture returns
one owned lease, with no primitive gates. A delegated cost provider's
exception passes through unchanged, as in the locked source for atomic
Allocate. Native cost delegation uses an explicit child declaration.
Generic resource delegation and actual foreign library consumers are closed below.
Atomic undeclared-child failures retain the original source behavior.

## Actual foreign wrapper and local-output lifetime closure

`test_always_source_binding.py` captures the actual `_always_and` example through
`import_register_emitter`, with `target` declared as caller-owned clean local
output. Forward and the independently captured source adjoint both survive
native snapshot JSON restoration. Their bound streams match the existing native
And compute and measured erase for every operand and feedback condition. No
foreign wrapper object reaches execution.

The native Always wrapper receives these owned directional emitters. Its shaped
inherited-control tuple is returned by identity and remains absent from the
child stream. A coherent spectator shares arbitrary amplitudes with the two
logical inputs. The entire source isometry agrees through `circuit.append(op)`.
Each erase outcome restores the complete unnormalized input and returns the
local target to zero. The forward emits 6 CX, 2 H, 2 T, 2 Tdg and 1 S. The
adjoint emits H, measurement, conditional CZ and reset. Independent source
Always(And) QEC refusal and child-level models are reused from the accepted
original test suite and retain their distinct meanings.

The actual Always(Add) wrapper with two- and three-bit words is captured in
both directions, then rebound to reversed logical words, one spectator and two
clean work slots. The direct unmodified source traversal independently matches
the instruction order/dependencies, operands, measurements and corrections.
Both directions use the ordinary append-to-IR entry and JSON restoration. Every
one of four measurement histories preserves the complete source complex action
with zero work population. All sixteen history pairs per starting direction
restore the full unnormalized input. Each direction emits 20 CX, 6 H, 4 T,
4 Tdg, 2 S, 2 measurements, 2 conditional CZ and 2 resets. The accepted native
Add resource declaration and source query remain independent of that inventory.

Source signature forwarding, actual factory/DocSpec identity, child display,
adjoint swapping and actual typed control adders have complete concrete
interface evidence. The generic resource and actual library closure below
completes Q-0314. The source atomic-child resource failure and original
serialization xfail remain recorded.

The final concentrated batch passed 142 cases with no warnings or skips.
Formatting, lint, strict typing for the changed Python file, Compiler scope,
architecture, multilevel IR and protected API checks all passed. Logs are
retained under `ftqc-reference-env/g2-always-binding-*.log`.


## Complete generic child delegation and foreign consumers

The internal resource owner `_component_trace/always.py` retains a hashable
native child independently of physical emitters. `AlwaysCall.callee_counts`
delegates to that child's direct declaration through the existing shared
accumulator and passes the same symbol allocator. Nested wrappers do not add
an artificial wrapper-to-child edge. Duplicate entries accumulate in encounter
order. Default adjoints restore the original identity after two inversions,
and specialized child inverses can use the existing inverse callback.

Unmodified plain/controlled Swap widths 1, 3 and symbolic n match both direct
edges and the complete independent QEC category model. The existing generic
cost query consumes the delegated graph with standard leaf costs. The original
`TestBloqWithCallGraph` fixture verifies three successive allocator states
through nested wrappers. Original Always(X), Always(And) and Always(Allocate)
direct graph queries fail through their atomic children. Native providers
propagate the exact exception object. A typed bodyless child can have an
explicit terminal model while physical binding requires a body. Replacing its
body with the existing native bit-gate function enables execution through the
same stored call.

Actual source libraries for Always(X/H) at depths 0, 1 and 2 preserve exact
bytes and reconstruct equal upstream. Existing explicit factories resolve
the already-bound native child once and return only the wrapper definition,
avoiding duplicate child registration. Reversed binding with an entangled
spectator matches all complex amplitudes; two appends restore the full input.
The original `_always_and` and Add captures, typed controls, measured erasure
histories, clean workspace and independent child models above remain accepted.
Actual serialization success is separate from the original harness's expected
serialization failure. No arbitrary foreign object enters native execution.

The new resource wrapper adds zero physical primitive gates, measurements or
work qubits. Primitive inventories above remain separate from graph estimates.
The complete source parent Q-0314 is accepted with original atomic declaration
failures retained, consistent with the unmodified source implementation.

Validation receipts `g2-always-generic-*` retain 233 concentrated and 252 shared
graph/control regressions, 17 integration cases and 21 Cirq 1.6.1 cases, all
passing. The unchanged original non-notebook suite has 5 passes and its original
serialization xfail, with 2 collection warnings. Shared graph tests retain 9
upstream parser/collection warnings. Strict typing for both new Python files,
Black, Ruff, team scope, architecture, multilevel IR and protected API checks
pass. Frozen source blobs and unchanged execution owners are independently
verified in the batch evidence receipt.

## Complete Always catalog dependency graph, 2026-11-22

The actual Always(And()) catalog factory now retains its existing native
wrapper Program and ordered And child. Its fresh graph and drawing match the
unchanged source. Two native-only rebinding helper interfaces are explicit.
Selected catalog execution through circuit.append(op) matches the complete
original tensor isometry. The RIGHT target remains a logical output.

Actual primitives are 6 CX, 2 H, 2 T, 2 Tdg, and 1 S. One logical target is
allocated at instruction offset zero, with peak three wires and no forward
measurements. The original root QEC query raises DecomposeTypeError because
Always delegates directly to atomic And.build_call_graph. The independent
original And child model declares one And and four T-equivalents. The refusal,
child estimate, and actual primitive histogram are retained separately.

The complete shared batch evidence, eight original atomic factory refusals,
source hashes and helper inventory are recorded in
[AUTO_PARTITION_FILE_AUDIT.md](AUTO_PARTITION_FILE_AUDIT.md) and the external
g2-bookkeeping-ui-* receipts. Existing measured erasure histories, clean
temporary workspace and exact recovery tests pass again. Existing generic
and serialization source boundaries remain explicit.
