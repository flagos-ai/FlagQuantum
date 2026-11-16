# Generic Trotter product file audit

Q-0449–Q-0451 cover the original generic product tutorial, implementation
and test file from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The three Git blobs matched, and the unchanged 10-cell source notebook
passed in an independent kernel. The original non-notebook tests returned
14 passes, one skip and one expected serialization failure.

The product accepts an ordered term library, an index sequence, a
coefficient sequence and a timestep. Each available pair contributes
one call at `2 * coefficient * timestep` in input order. Python negative
indices and truncation at the shorter sequence remain covered. The
native term protocol exposes angle-bound emission and resource dispatch.
Terms can declare typed signatures; existing terms with only a layout
infer an opaque or bit through signature from their port widths.

The product verifies full signature equality and matching physical
layout before scheduling. Shaped through ports retain their encoding
and shape. Same-width shape or encoding mismatches, non-through
ports and declarations that disagree with the physical layout are
rejected. The emitted instruction stream is checked for use of
undeclared wires.

The three-site second-order example uses X–ZZ–X at timestep 0.01 with
coefficients 0.05, 2 and 0.05. Its immediate graph contains two X calls
and one ZZ call. Native execution emits six RX, three RZ and six CX.
The declared model gives nine rotations and a three-qubit peak. It
uses no scratch or feedback. The exact fixture's complex matrix and
one-level call graph match the frozen example; the runnable native
notebook also checks the adjoint and renders both diagrams.

The existing round-trip evidence covers second- and fourth-order
Suzuki parameters, asymmetric schedules and symbolic angle binding.
The native symbolic emitted CircuitIR also round-trips through JSON.
The original expected-failure case concerns serialization of the
source bloq example. Native serialization is at the emitted IR boundary.
