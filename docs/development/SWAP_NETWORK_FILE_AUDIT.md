# Swap network source audit

The locked reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The eight source and asset blobs
are pinned in `tests/unit/test_swap_network_file.py`.

| Source export | Native ownership and execution |
| --- | --- |
| CSwapApprox | Compiler `relative_controlled_swap_ir` keeps the literal relative-phase word exchange and shared fanout. The Simulation raw controlled swap owns its basis permutation. The resource declaration retains T and one- and two-wire Cliffords. |
| MultiplexedCSwap | Compiler `multiplexed_cswap_ir` uses bounded unary traversal with exact Fredkin leaves. Selection words are big-endian and targets are row-major. The output is an arbitrary through word. Coherent leaves use the live traversal predicate. |
| SwapWithZero | Compiler `swap_with_zero_ir` uses the source multi-axis schedule, traversing the innermost axis first. The inverse reverses the ordered stages. The leaf is the same relative-phase exchange as CSwapApprox. |

The exported names have named native owners. The examples bind them without
introducing a second package-level API. The [native
notebook](../../examples/notebooks/swap_network.ipynb) retains all three source
sections and their original factory dimensions, including both multidimensional
SwapWithZero shapes and the symbolic declaration. It shows the signatures and
depth-one call graphs and appends each concrete operation through
`circuit.append(op)`.

## Verification

- The approximate swap checks the four- and 64-bit factories, the bare symbolic
  factory, and the original small and large count grids. The four-bit coherent
  test compares all complex amplitudes with a spectator and verifies the inverse
  roundtrip. An unconstrained width retains the source fallback
  `Max(2, 2*n + 1)` through the shared resource dispatcher.
- The multiplexed swap checks all original size triples and the five-row
  factory. The gate comparison retains source per-wire dependencies, exact
  operands and measured feedback in both directions. The sparse execution
  compares the complex selection and data output with clean traversal work.
  Each large case uses three histories. The existing small tests enumerate
  every measurement history and include padded addresses.
- SwapWithZero checks the original basis, diagram, resource and factory
  dimensions, including the seeded multi-axis resource fixture. Both directions
  match the source stage traversal and leaf streams. Smaller cases compare the
  entire complex map and roundtrip. The classical schedule is compared with
  the original in-place callback for all addresses in both original data grids.
  The symbolic model retains one fewer exchange than the product of axis
  lengths.
- The unchanged upstream suite, including its notebook, passed 64 cases with
  five skips and two expected failures. The skips are symbolic decomposition
  or dtype checks. The expected failures are upstream counts enforcement flags.

| Original fixture | Emitted T/T† | Emitted CX | Emitted measures | Reserved wires | Source declared categories | Source logical wires |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Four-bit CSwapApprox | 16 | 23 | 0 | 9 | 16 T, 87 Clifford | 9 |
| Four 32-bit rows, SwapWithZero | 384 | 573 | 0 | 136 | 384 T, 2109 Clifford | 136 |
| Five two-bit rows, MultiplexedCSwap | 82 | 101 | 3 | 17 | 10 CSwap, 3 And, 8 Clifford, 3 measurements | 17 |

MultiplexedCSwap's emitted T count uses seven T-family gates per exact Fredkin.
The source model retains the CSwap count and can convert it using its separate
default equivalence. Reserved work and live-width estimates are checked
independently in padded cases.

## Remaining boundaries

- Q-0707 retains symbolic execution and the upstream abstract Clifford cost
  query for generic controls.
- Q-0709 retains source singleton and zero-width selection failures and the
  generic control cases requiring unresolved controlled And leaves.
- Q-0712 retains executable symbolic binding, source earlier zero-bit axis
  rejection, and a native in-place array callback. The current compiler
  owns the shared stage permutation and the quantum emitter.
