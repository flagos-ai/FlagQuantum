# Reflection using preparation

The register emitter binds the [locked source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/reflections/reflection_using_prepare.py)
in its original order: inverse preparation, zero-selective phase,
scalar phase, then preparation. The supplied circuits must be an
inverse pair independent of the outer phase target. They own their
own preparation registers. The zero predicate checks the selection
register only. A caller with logical preparation junk retains its
existing contract. The reflection restores all its borrowed work.

The phase target is the caller's control for a controlled reflection.
An open control uses the original X sandwich. For an uncontrolled
reflection a clean anchor holds one inside the same sandwich. The
source's zero-qubit global phase is represented by a one-qubit phase
on this anchor. This preserves the scalar phase and returns the
anchor to zero. The controlled source phase uses the same phase gate.

MultiControlZ uses the source compute/CZ/erase sequence. It shares
the prefix implementation with MultiControlX. For a selection width
of three the reflection uses 8 T-family gates and two measurements.
The uncontrolled case borrows three qubits including the phase
anchor, and the controlled case borrows two. These counts match the
source MultiControlZ model. The optional scalar phase is counted
separately. For example, the three-state uniform preparation on a
two-qubit selection gives 44 emitted T-family gates, 11 measurements
and four preparation Rz rotations for one reflected application.

The locked `adjoint()` returns the same object for every coefficient.
For coefficient i, multiplying that source matrix by its declared
adjoint gives minus identity. The native inverse conjugates the
coefficient and is tested against the unchanged source instantiated
with that conjugate. The original behavior remains a reference test.

Tests check complete gate order and resource counts for the source
identity-preparation circuit, including empty selection and both
control polarities. The nontrivial path composes the independently
verified uniform preparation and checks its complete reflection on
complex inputs, measured cleanup, and forward/inverse restoration.
