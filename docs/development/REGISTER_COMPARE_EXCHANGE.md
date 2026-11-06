# Shared compare/exchange register primitives

Run `python -m examples.algorithms.register_compare_exchange` without Qualtran or
NumPy. `circuit.append(OffsetComparisons())` compares the original offset-two
pairs in five one-bit words, retaining two logical history bits and leaving the
trailing word untouched. Two separate comparison work bits are zero afterward.
Appending the original adjoint restores all words and clears both histories.
Full coherent complex inputs and both results are checked through the existing
native dynamic runtime.

## Ownership and interfaces

All expansion lives in compiler/registers and returns existing Instruction
sequences. No execution backend, implicit runtime allocation, new representation
or protected root API is introduced. Words use big-endian unsigned bit patterns.
Classical slots retain latest random outcomes, not a full measurement event log.

- `ordering.greater_adjoint_ir(left,right,target,workspace,classical_bits)`:
  original X(target), then original self-adjoint LessThanEqual. Operands may
  have different positive widths; target can be arbitrary. Reuse the existing
  less_equal_requirements query and clean work implementation. Although the
  ideal XOR map is self-inverse, the source forward final X and inverse first
  X are kept in their actual component positions.
- `compare_exchange.compare_exchange_ir(left,right,history,workspace,
  classical_bits,adjoint=False)`: equal positive widths. Forward history starts
  zero, then GreaterThan computes a>b and the original exact CSwap exchanges
  the words conditionally. Outputs are min/max and that history. The work pool
  is clean immediately; history is logical output and must be retained. Inverse
  requires a correct complete forward output tuple, performs original CSwap,
  then GreaterThan adjoint, and clears history. It does not reset an arbitrary
  dirty history or discard sorting information.
- `parallel_comparison.parallel_compare_ir(words,offset,histories,workspace,
  classical_bits,adjoint=False)`: equal positive word widths, positive count
  and offset. `parallel_comparison_pairs` follows original block/step order and
  the early break in incomplete trailing blocks. Each pair has a distinct
  logical history; inverse consumes correct histories in reverse source order.
  One comparison pool and its slots are reused only after each child cleans
  work. Zero-pair blocks use empty work/histories/slots and emit nothing, matching
  the original empty component graph. The name denotes independent logical
  pairs, not simultaneous execution or a distributed performance claim.

Responsibilities are separate: ordering owns comparison, controlled_swap owns
the exact bit/register exchange, compare_exchange composes those two, and
parallel_comparison only traverses source pairs. It contains no copied tree,
And network, swap synthesis, runtime or full sorting algorithm.

For one n-bit pair, supply 3n-1 zero work wires and 2n-1 classical slots. One
additional history wire is logical output. For r offset pairs, retain r history
wires but reuse the same work/slot budgets. Other unpaired words remain intact.

## Unmodified source reference and port binding

The reference is Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389: arithmetic/comparison.py::GreaterThan
and LessThanEqual, arithmetic/sorting.py::Comparator/ParallelComparators, and
the original basic_gates/swap.py exact CSwap decomposition. Google Apache-2.0
attribution is retained in the owning modules.

Tests read original decompositions and compare complete gate multisets, ordered
per-wire dependencies, operands and measurement feedback. Parallel source tests
expand only original Comparator nodes once, retaining whole GreaterThan/CSwap
component lifetimes and original RIGHT/LEFT history ports. Unrestricted early
flattening changes temporary allocation order and is not used to infer native
wire bindings. The existing source adapter keeps GreaterThan's components before
Cirq export; unequal-width comparisons use its existing allocation-order mode.
These are temporary-name/port bindings, not changes to source algorithms.

The original CSwap and LessThanEqual adjoints explicitly return themselves.
Comparator inverse still follows its original reversed component chain. An
invalid inverse tuple (equal words with history=1) retains its nonzero history
in native IR; no reset is added to manufacture a clean inverse-domain claim.

## Complex channels and clean work evidence

GreaterThan adjoint is source-checked for sixteen equal/unequal width pairs
through eight bits. Small full complex inputs, arbitrary targets and entangled
spectators enumerate every measurement history. Forward restoration is checked.

Comparator source forward/inverse streams are compared at n=1/2/3/5/8. Every
pair of n-bit values through eight bits is checked with its logical min/max/
history and zero work. Correct full outputs restore original words and clear
history. At n=1/2/3, full raw complex spectator channels enumerate all 2/8/32
histories separately in each direction.

Parallel source forward/inverse tests cover seven count/offset/width settings,
including unequal trailing blocks and eight-bit pairs. All input lists with
at most eight logical data bits are exhausted in tested settings; larger lists
use explicit adversarial/equal/ascending/descending cases. Raw forward/inverse
spectator tests enumerate every history for the tested 4/1/1, 5/2/1 and 3/1/2
settings. Do not interpret those as exhaustive enumeration of every large list.

The strict compositional certificate independently recognizes literal And and
exact CSwap blocks, checks clean And inputs and both erasure outcomes, and proves
identical normalized complex maps for every history. Branches have amplitude
factor 2**(-m/2). Logical history bits remain part of the output state; only the
comparison pool is claimed clean immediately. This is not dense enumeration of
all 2**15 raw trajectories at n=8. Symbolic and broader sorting families remain
outside this concrete acceptance.

## Measured primitives and independent model estimates

T includes T-dagger. Conditional CZ below is an emitted feedback instruction;
actual execution depends on outcomes. Each measurement has one reset and S.

| Circuit | CX | X | T | H | Measurements / S / conditional CZ / reset | Logical histories | Clean work | Total qubits | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreaterThan adjoint, 8/8 | 225 | 22 | 60 | 45 | 15 | 0 | 23 | 40 | 15 |
| Comparator, n=8, either direction | 289 | 22 | 116 | 61 | 15 | 1 | 23 | 40 | 15 |
| ParallelComparators, count=7, offset=2, n=3 | 297 | 36 | 123 | 63 | 15 | 3 | 8 | 32 | 5 |

The GreaterThan target is existing logical input/output, not an allocated history.
Independent pinned QECGatesCost gives one pair 8n-4 comparison T plus a CSwap
model. Its default four-T-per-bit exchange estimate is 12n-4; setting the original
`ts_per_cswap=7` gives 15n-4, matching the executed original seven-T source.
Thus n=8 Comparator is measured 116 T versus default estimate 92 T. For r pairs,
T and measurements multiply by r, while the work pool and stored slots do not.
The seven-word example has default estimate 96 T versus measured/exact-model
123 T. Source QubitCount independently matches the stated layouts.

Abstract source fields remain separate: GreaterThan/Comparator at n=8 have
and_bloq=15, clifford=168, measurement=15, with cswap=8 for Comparator. The
seven-word example has and_bloq=15, clifford=174, measurement=15, cswap=9.
These are not primitive Clifford inventories. The default/exact T difference is
a model weight convention, not a native synthesis penalty or lost phase.
