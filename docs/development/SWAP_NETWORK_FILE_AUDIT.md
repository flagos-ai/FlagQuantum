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
  entire complex map and roundtrip. The native in-place callback reuses that
  stage order and is compared against every address in both original data grids.
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

## Additional control on the relative swap

A scalar positive outer control directly declares controlled T and
controlled one- and two-wire ArbitraryClifford children. The counts are
`4*n`, `16*n` and `6*n - 1`. The adjoint retains the source directions
on all three children. A negative scalar control adds two X calls around
the positive-controlled swap. A shaped single-bit control declares the
positive child even when its value is one. With multiple outer controls,
the wrapper declares predicate compute, the positive child and predicate
erase. The outer signature preserves the scalar versus shaped distinction.

The append example uses the existing controlled emitter. Width-one and
width-two fixtures compare operands, gate dependencies, feedback slots
and peak width in both directions. The broader controlled exchange suite
compares full complex amplitudes and all small measurement histories,
including coherent inactive sectors and clean predicate work. The source
aggregate QEC query still fails on controlled ArbitraryClifford. Concrete
emitted counts remain available independently. For a two-bit swap with
two outer controls they include 11 CCX, 16 CH, 24 controlled phases,
four T/T† in the predicate and one measurement.

## Multiplexed swap register layout

The typed layout preserves each ordered bounded unsigned selector and
derives the target array axes from its iteration limit. Control registers
retain their shape. Target and output words stay opaque at width one,
matching the source. Selectors are big endian, and the target wires are
partitioned in row-major order. The layout supplies the concrete append
binding and the resource model with the same dimensions.

The forward and adjoint grid verifies selectors with padding, multiple
axes, several controls and singleton axes with a live control. It compares
the typed ports, primitive dependencies, all small measurement histories,
full complex output and clean work against the locked source. The
symbolic fixture verifies the declared target axes and shaped controls.
Physical binding requires concrete dimensions. Reserved wires and the
source live width remain distinct when selectors include padding.

## SwapWithZero typed and classical bindings

The compiler declares every live selector as bounded unsigned with its
original axis extent. The target port retains the full array shape,
including omitted zero-bit selection axes. Both execution routes use
the same stage traversal. The append route uses physical MSB-first
selection wires and relative-phase exchanges. The classical callback
uses integer selection bits and mutates the caller's array directly,
without changing selection objects. The adjoint reverses the stage order.

The array checks include strided views and preserve the locked callback's
keyword insertion order. A trailing zero-bit axis works. When a leading
zero-bit axis precedes an exchange, the raw source and native callback
raise `IndexError`. The compiler reports that boundary before emitting
instructions.

## Remaining boundaries

- Q-0707 retains symbolic execution and the upstream abstract Clifford cost
  query for generic controls.
- Q-0709 retains source singleton and zero-width selection failures and the
  generic control cases requiring unresolved controlled And leaves.
- Q-0712 retains executable symbolic binding and the earlier zero-bit
  axis failure. The compiler owns the stage permutation and the quantum
  emitter. Simulation owns the in-place data callback.
