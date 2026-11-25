"""Original OAA amplitudes and four equal-weight Pauli choices."""

AMPLITUDES = (
    0.470924371412456,
    0.813430359698115,
    0.000129158385783,
    0.341410705235337,
)
PAULI_TERMS = ("XX", "XZ", "ZX", "ZZ")
# Acting after X, this exact single-qubit unitary changes the choice to Z.
X_TO_Z = ((0, 1), (-1, 0))
