# Sparse rotation preparation

The sparse emitter preserves coefficient and index order, pads coefficients
with zeros to a power of two, and normalizes the dense word. It partitions the
big-endian target without gates, prepares the low word with the caller's phase
gradient, then applies the source basis permutation to the full target. The
work pool and classical slots are reused after the dense stage clears its
work. Empty cycles implement the identity without allocating permutation
scratch. A target register can be wider than the minimum width for the state.

The shared partial-map constructor follows the locked source's iteration
order, including its behavior when a path reaches a previously visited
vertex. For example, indices (3, 0) request 0 → 3 and 1 → 0, but the locked
code emits only the cycle (0, 3). The sparse emitter checks the original
edges and rejects this input. It also requires at least two coefficients
because the locked singleton composition fails to bind a target register.

For indices (2, 1), coefficients (1+i, 1-i), state size 3, a four-qubit target,
and an eight-qubit gradient:

| Evidence | Value |
|---|---:|
| Native T/T† | 100 |
| Native CX | 241 |
| Native measurements | 25 |
| Native work qubits | 16 |
| Native reused classical slots | 7 |
| Native total qubits | 28 |
| Source model Toffoli | 12 |
| Source model And computes | 11 |
| Source model measurements | 11 |
| Source model total qubits | 22 |

The gradient-add boundary and its different source cost model are documented
in [dense rotation preparation](./REGISTER_ROTATION_PREPARATION.md). Tests
read the original partition ports and permutation cycles, compare the
executable component streams, and compare complex outputs with the original
source tensors. The small cases enumerate all histories with at most five
measurement events. The eight-bit witness verifies three histories and all
amplitudes, including work cleanup and the returned gradient, against a
two-bit source tensor after checking exact scaling of every angle word.

- [Locked partial map](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/permutation.py)
- [Locked sparse preparation](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/sparse_state_preparation_via_rotations.py)
