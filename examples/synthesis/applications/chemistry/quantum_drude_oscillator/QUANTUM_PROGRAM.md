# Drude oscillator quantum program

The native program retains all forty parameters, the ordered ten pair layers
and six output qubits. Quantum slices are checked views; explicit four-element
classical argument lists preserve the source's static parameter slices without
adding a second array representation or host execution during capture.

The existing `R(theta, pi/2, qubit)` expresses RY exactly. This makes the program
runnable today with public native APIs, but uses more rotations than a direct RY
marker. At the retained catalog parameters, native resources are 6 qubits,
112 operations and depth 30, versus original 6 qubits, 90 operations and depth 23.
Both use zero workspace. This is a stated resource gap, not resource parity.

Acceptance compares all complex amplitudes against independent direct Circuit
RY/RZ/CX gates for three nonzero parameter sets, including negative angles and
angles beyond 2*pi. Bit order and absolute phase are included. The complete
energy evaluation, data and optimization Notebook remains pending; this
quantum-program counterpart does not certify it.
