# Retained amplitude preparation

`main.py` retains the source's full 64-element signed amplitude vector and 0.3
error upper bound, bound through a native classical real-array parameter. The
existing exact amplitude kernel satisfies this bound. This example does not
claim a new approximation algorithm or a lower resource count.

The independent Circuit uses a Householder reflection rather than the compiler's
rotation tree. Original and native outputs agree in every complex amplitude,
including signs, absolute phase and bit order. Both use 6 qubits, 125 operations,
depth 120 and no workspace. The probability-vector L2 error is below 1e-12.

The complete Notebook additionally investigates approximation and resource
tradeoffs. That workflow remains pending; only its retained quantum program is
certified here. Source data, identity and this scope are recorded in provenance.
