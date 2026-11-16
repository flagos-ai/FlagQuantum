# Native parameterized ansatz composition

The retained quantum program prepares four bits and applies one XX generator
with the first real-array parameter. Native qfunc expresses the layer and
ansatz as named helpers; the compiler owns gate decomposition and bit placement.
`build_program(theta)` supplies the source's one-element parameter array.
The plain Python list annotation describes a bounded native real array; unlike
the source fixed-array declaration it is not a static length-one annotation.

The native and original programs both use four allocated bits, seven operations,
depth five and no workspace. Independent Circuit XX rotations check two nonzero
angles under both synthesis objectives, including absolute complex phase.

This is the complete retained quantum program. The source Notebook additionally
implements operator-pool screening, ansatz growth, optimization and plots. Those
ADAPT-VQE training workflows are still pending and are not counted here.
