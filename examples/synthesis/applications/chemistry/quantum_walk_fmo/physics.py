"""Seven-site FMO Hamiltonian and its declared 35-term product-formula order."""

import flagquantum as fq

# Wavenumber units (cm^-1); the eighth computational state pads this matrix by zero.
SITE_HAMILTONIAN = (
    (280, -106, 8, -5, 6, -8, -4),
    (-106, 420, 28, 6, 2, 13, 1),
    (8, 28, 0, -62, -1, -9, 17),
    (-5, 6, -62, 175, -70, -19, -57),
    (6, 2, -1, -70, 320, 40, -2),
    (-8, 13, -9, -19, 40, 360, 32),
    (-4, 1, 17, -57, -2, 32, 260),
)

# Indices bind logical target elements; zero is the least significant bit.
HAMILTONIAN = (
    226.875 * (fq.I())
    + -32.0 * (fq.X(0))
    + 13.75 * (fq.X(0) @ fq.X(1))
    + -19.5 * (fq.X(0) @ fq.X(1) @ fq.X(2))
    + -2.25 * (fq.X(0) @ fq.X(1) @ fq.Z(2))
    + -20.0 * (fq.X(0) @ fq.Y(1) @ fq.Y(2))
    + -1.0 * (fq.X(0) @ fq.Z(1))
    + 12.75 * (fq.X(0) @ fq.Z(1) @ fq.X(2))
    + -21.0 * (fq.X(0) @ fq.Z(1) @ fq.Z(2))
    + -15.75 * (fq.X(0) @ fq.X(2))
    + -52.0 * (fq.X(0) @ fq.Z(2))
    + -15.0 * (fq.Y(0) @ fq.X(1) @ fq.Y(2))
    + 16.25 * (fq.Y(0) @ fq.Y(1))
    + 15.5 * (fq.Y(0) @ fq.Y(1) @ fq.X(2))
    + 0.25 * (fq.Y(0) @ fq.Y(1) @ fq.Z(2))
    + 16.75 * (fq.Y(0) @ fq.Z(1) @ fq.Y(2))
    + -11.75 * (fq.Y(0) @ fq.Y(2))
    + -11.875 * (fq.Z(0))
    + 3.5 * (fq.Z(0) @ fq.X(1) @ fq.X(2))
    + 1.0 * (fq.Z(0) @ fq.X(1) @ fq.Z(2))
    + 5.5 * (fq.Z(0) @ fq.Y(1) @ fq.Y(2))
    + -33.125 * (fq.Z(0) @ fq.Z(1))
    + -6.0 * (fq.Z(0) @ fq.Z(1) @ fq.X(2))
    + 41.875 * (fq.Z(0) @ fq.Z(1) @ fq.Z(2))
    + 2.5 * (fq.Z(0) @ fq.X(2))
    + -66.875 * (fq.Z(0) @ fq.Z(2))
    + 3.0 * (fq.X(1))
    + -6.0 * (fq.X(1) @ fq.X(2))
    + 4.0 * (fq.X(1) @ fq.Z(2))
    + -4.0 * (fq.Y(1) @ fq.Y(2))
    + 118.125 * (fq.Z(1))
    + 0.5 * (fq.Z(1) @ fq.X(2))
    + 13.125 * (fq.Z(1) @ fq.Z(2))
    + 9.0 * (fq.X(2))
    + -8.125 * (fq.Z(2))
)

# Matrix words use the most significant factor first. Order is significant.
PAULI_ORDER = (
    "III",
    "IIX",
    "IXX",
    "XXX",
    "ZXX",
    "YYX",
    "IZX",
    "XZX",
    "ZZX",
    "XIX",
    "ZIX",
    "YXY",
    "IYY",
    "XYY",
    "ZYY",
    "YZY",
    "YIY",
    "IIZ",
    "XXZ",
    "ZXZ",
    "YYZ",
    "IZZ",
    "XZZ",
    "ZZZ",
    "XIZ",
    "ZIZ",
    "IXI",
    "XXI",
    "ZXI",
    "YYI",
    "IZI",
    "XZI",
    "ZZI",
    "XII",
    "ZII",
)
