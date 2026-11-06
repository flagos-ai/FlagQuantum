# Multiply two unsigned fixed-point fractions

The native emitter binds the pinned Qualtran v0.7.0
MultiplyTwoReals signature. All three words have the same width
and contain only fractional bits. The forward result starts at
zero. The adjoint erases the corresponding computed result.

Appendix D5 truncates each shifted row before adding it. The
lowest multiplier bit contributes zero, the next bit controls a
one-bit copy, and the remaining bits control progressively wider
prefix additions. The inverse reverses the additions and erases
the copy by measurement. Integer scaling uses the same emitter
and preserves its original literal gate stream.
[Appendix D5](https://arxiv.org/pdf/2007.07391#page=71)

With three-bit words 7 and 7, term truncation produces output
word 4, while shifting the complete product produces 6. Tests
compare the original fixed-point signature and resource formula,
the pinned CAdd and CAdd adjoint child decompositions, and the
independent term formula for every input at widths 1, 2, 3, 4
and 6. Exact And certificates cover every measurement history.
Raw projection for widths 1 and 2 and sparse primitive
trajectories for widths 3 and 4 check full complex amplitudes
with a spectator and verify cleanup. A two-bit round trip
exercises `circuit.append(op)` on the local trajectory executor.

The source member has no executable gate, tensor or classical
implementation. It provides the signature and the forward
`d*d-d-1` Toffoli formula. Width one is an all-zero product in
the executable approximation and the native stream is empty.
The pinned model returns minus one at that width.

| Width | Native forward T-family / CX / measurements | Native erase T-family / CX / measurements | Work wires | Pinned Toffoli model |
| ---: | --- | --- | ---: | ---: |
| 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | -1 |
| 2 | 4 / 6 / 0 | 0 / 0 / 1 | 0 | 1 |
| 4 | 44 / 88 / 10 | 40 / 82 / 11 | 6 | 11 |
| 6 | 116 / 242 / 28 | 112 / 236 / 29 | 10 | 29 |
| 10 | 356 / 766 / 88 | 352 / 760 / 89 | 18 | 89 |
