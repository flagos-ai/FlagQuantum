# Bitonic register networks with original histories

Run `python -m examples.algorithms.register_bitonic_sort` without Qualtran or
NumPy. Users write `circuit.append(BitonicWords(operation, inverse))`. The
four-word example runs merge and sort in both original directions on complex
coherent inputs, verifies two clean comparison work bits, and restores complete
words/history with the opposite direction. Histories remain logical output
until the correct original adjoint clears them.

## Separate ownership and interfaces

These are the Q02 register-network components from the pinned sorting module.
No runtime backend, protected root API or additional IR representation is added.

- `comparison_layout.py` owns comparison_network_requirements(width,
  history_count) and shared physical validation. One comparison pool has 3n-1
  clean work wires / 2n-1 stored classical slots, independent of the number of
  histories. Zero comparisons use no work or slots. Histories are excluded from
  clean work. The existing parallel_comparison consumer now uses this same
  validator; its original gate streams remain in regression.
- `bitonic_merge.py` owns bitonic_merge_ir(words,histories,workspace,
  classical_bits,adjoint=False) and bitonic_merge_history_count(half_length).
  Each half has positive power-of-two length h and is sorted ascending for the
  merge guarantee. Words have equal positive big-endian unsigned widths.
  Source result ports are the original xs/ys physical words concatenated, with
  no extra wire exchanges. First compare i with 2h-1-i, then call the existing
  offset comparison blocks in descending log-offset order. History is first
  cross-round then each offset round, totaling h*(log2(h)+1) bits.
- `bitonic_sort.py` owns bitonic_sort_ir with the same physical arguments and
  bitonic_sort_history_count(count). Count k is a positive power of two.
  Preserve source k=1 empty and k=2 Comparator cases, then original left sort,
  right sort and merge. Concatenate histories left/right/merge, totaling
  (k//2)*log2(k)*(log2(k)+1)//2 bits. The original adjoint reverses components,
  restoring arbitrary original word order and clearing correct histories.

Both networks reuse existing compare_exchange and parallel_compare emitters;
there is no copied comparison tree or swap synthesis. Each component cleans its
work before the same physical pool is reused. Classical slots retain latest
outcomes, not the full event history. Forward histories start zero and become
logical outputs entangled with words. Inverse inputs must be correct complete
forward outputs; arbitrary histories are not reset to manufacture cleanup.

An unsorted-half input still executes the original merge network, but need not
be globally sorted. The native counterexample (1,2,3,0) agrees with the unchanged
source. Non-power-of-two lists are explicitly refused, retaining the source's
supported-domain restriction. Single-word sorting matches its empty graph.

## Original source calls, gates and ports

The reference is unmodified Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389, arithmetic/sorting.py and its original
BitonicMerge/BitonicSort adjoint objects. Google Apache-2.0 attribution is
retained. Primitive comparison and exact CSwap remain the previously pinned
comparison.py/swap.py implementations.

The reference test walks original hierarchical CompositeBloq/Soquet connections.
It reads original child calls and bindings, including LEFT/RIGHT word/history
ports, and obtains every leaf gate from the unchanged Comparator decomposition.
Separate source-private work pools are bound to the native reusable clean pool
only at complete component boundaries. Output connections are checked, not
inferred from a copied sorting algorithm.

Unrestricted global flattening greedily interleaves independent branches and
changes temporary birth order. That is a different legal schedule; it is not
used as a wire-allocation oracle. Tests preserve original hierarchical component
lifetimes, then compare full primitive multisets, ordered per-wire dependencies,
operands and feedback without weakening the existing equality assertion. No
upstream source is patched or replaced with native algorithm logic.

## Complex amplitudes, histories and work

Source gate/port/model tests cover seven merge and eight sort parameter settings,
both directions, including three-bit eight-word networks and eight-bit pairs.
The strict literal-And/exact-CSwap certificate checks all erased targets and
clean work on each tested basis input, proving every measurement history has
its full complex map times 2**(-m/2). Sorting exhausts every input list for tested
settings with at most eight logical data bits; merging exhausts the sorted-half
promise in those settings. Larger settings use explicit equal and sorted-half/
ascending/descending adversarial samples, not every possible large list.

Raw complete complex spectator channels use actual unnormalized projections in
both directions. Expected logical histories are obtained from the unchanged
source classical execution. Merge enumerates every 2/16/8 raw history for the
(half,width)=(1,1),(2,1),(1,2) cases; sort enumerates every 2/64/8 for
(count,width)=(2,1),(4,1),(2,2). No dense 2**120 trajectory or large statevector
claim is made. Original adjoint restoration checks complete data and histories;
only the comparison pool is declared zero immediately after forward execution.

## Measured primitives and independent source estimates

At eight three-bit words, T includes T-dagger. Conditional CZ counts are emitted
feedback instructions, whose execution depends on outcomes. S/reset counts equal
measurement counts. Classical slots are reused across source components.

| Network, either direction | Comparisons / logical histories | CX | X | T | H | Measurements / S / conditional CZ / reset | Clean work | Total qubits | Stored classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BitonicMerge, half=4,n=3 | 12 | 1,188 | 144 | 492 | 252 | 60 | 8 | 44 | 5 |
| BitonicSort, count=8,n=3 | 24 | 2,376 | 288 | 984 | 504 | 120 | 8 | 56 | 5 |

Independent QECGatesCost with original ts_per_cswap=7 gives r*(15n-4) T and
r*(2n-1) measurements, matching measured source primitives. Its default
four-T-CSwap estimates are 384 and 768 T; the exact measured estimates are
492 and 984 T. The differences 108/216 are 3*n*r from the original cost weight,
not a native synthesis penalty. QubitCount independently matches tested layouts.

Abstract source fields remain separate from primitive Clifford inventories:
merge has and_bloq=60, clifford=696, cswap=36, measurement=60; sort has
and_bloq=120, clifford=1392, cswap=72, measurement=120. This does not claim
simultaneous or distributed execution. Symbolic parameters and remaining source
file presentation/contracts stay outside this concrete network acceptance.
