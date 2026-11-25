"""Independent CNF truth and exact Circuit product-state reference."""

from collections.abc import Sequence

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.main import (
    build_program,
)
from flagquantum.simulation.mps.state import MPSState

CLAUSES = (
    (0, 29, 48),
    (1, 31, 66),
    (2, 3, 6),
    (3, 16, 56),
    (4, 47, 65),
    (5, 24, 47),
    (6, 14, 2),
    (7, 55, 5),
    (8, 56, 37),
    (9, 60, 31),
    (10, 34, 51),
    (11, 13, 58),
    (12, 64, 0),
    (13, 57, 25),
    (14, 40, 33),
    (15, 17, 50),
    (16, 33, 54),
    (17, 53, 7),
    (18, 26, 28),
    (19, 39, 52),
    (20, 1, 40),
    (21, 65, 30),
    (22, 21, 36),
    (23, 28, 46),
    (24, 62, 32),
    (25, 19, 59),
    (26, 61, 45),
    (27, 42, 20),
    (28, 8, 64),
    (29, 22, 13),
    (30, 27, 34),
    (31, 38, 15),
    (32, 52, 22),
    (33, 7, 18),
    (34, 4, 11),
    (35, 43, 16),
    (36, 58, 4),
    (37, 11, 1),
    (38, 66, 10),
    (39, 63, 43),
    (40, 48, 62),
    (41, 10, 8),
    (42, 35, 14),
    (43, 50, 19),
    (44, 51, 55),
    (45, 2, 24),
    (46, 59, 29),
    (47, 37, 21),
    (48, 41, 44),
    (49, 6, 17),
    (50, 20, 42),
    (51, 0, 49),
    (52, 54, 23),
    (53, 46, 9),
    (54, 23, 26),
    (55, 18, 39),
    (56, 44, 41),
    (57, 45, 61),
    (58, 25, 12),
    (59, 49, 63),
    (60, 15, 3),
    (61, 32, 27),
    (62, 12, 53),
    (63, 30, 60),
    (64, 9, 38),
    (65, 36, 35),
    (66, 5, 57),
)


def marked(enabled: tuple[int, ...]) -> bool:
    return all(a in enabled or b not in enabled or c in enabled for a, b, c in CLAUSES)


def reference_circuit(enabled: tuple[int, ...] = ()) -> fq.Circuit:
    circuit = fq.Circuit(150, dtype=torch.complex128)
    for bit in enabled:
        circuit.x(66 - bit)
    if marked(enabled):
        # The basis oracle is a global minus sign, with every wire restored.
        circuit.x(149).z(149).x(149)
    return circuit


def product_signature(
    tensors: Sequence[torch.Tensor],
) -> tuple[tuple[int, ...], complex]:
    bits = []
    phase = 1 + 0j
    for site in tensors:
        assert site.numel() == 2
        zero, one = (complex(value) for value in site.reshape(2))
        assert (zero == 0) != (one == 0)
        bits.append(int(zero == 0))
        phase *= one if zero == 0 else zero
    return tuple(bits), phase


def compare() -> None:
    program = build_program()
    actual = fq.run(program).native()
    expected = fq.run(
        reference_circuit().to_ir(),
        options=fq.ExecutionOptions(
            mode="mps",
            backend="torch",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
            memory_limit_bytes=4800,
        ),
    ).native()
    assert isinstance(actual, MPSState) and isinstance(expected, MPSState)
    assert not actual.truncation_records and actual.max_bond == 1
    assert product_signature(actual.tensors) == product_signature(expected.tensors)
    bits, phase = product_signature(actual.tensors)
    assert bits == (0,) * 150 and phase == -1
    print(
        "All 150 product-state bits, absolute minus sign and 83 clean work bits agree."
    )


if __name__ == "__main__":
    compare()
