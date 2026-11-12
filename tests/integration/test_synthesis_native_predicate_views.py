"""Unsigned numeric bit views retain coherent control and checked borrowing."""

import math
from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("marker", [0, 1, 2])
def test_slice_predicate_matches_independent_coherent_circuit(marker: int) -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], flag:Output[QBit]",
            "allocate(x)\nallocate(flag)\nhadamard_transform(x)\n"
            + ("X(flag)\n" if marker == 1 else "")
            + (
                "R(0.73, 1.5707963267948966, flag)\nRZ(0.41, flag)\n"
                if marker == 2
                else ""
            )
            + "control(x[1:4] == 2, lambda: X(flag))",
        ),
    )
    actual = fq.run(program).statevector().reshape(32, -1)
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    if marker == 1:
        circuit.x(4)
    elif marker == 2:
        circuit.ry(4, 0.73).rz(4, 0.41)
    permutation = torch.eye(32, dtype=torch.complex128)
    for x in (4, 5):
        permutation[[2 * x, 2 * x + 1]] = permutation[[2 * x + 1, 2 * x]]
    circuit.any(*range(5), matrix=permutation)
    torch.testing.assert_close(
        actual[:, 0], circuit.state().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24


def test_disjoint_and_shared_slices_use_little_endian_numeric_values() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], y:Output[QNum[3]], flag:Output[QBit]",
            "allocate(x)\nallocate(y)\nallocate(flag)\n"
            "hadamard_transform(x)\nhadamard_transform(y)\n"
            "control((x[1:4] == y[:3]) & (x[2] == 1), lambda: X(flag))",
        )
    )
    actual = fq.run(program).statevector().reshape(256, -1)
    expected = torch.zeros(256, dtype=torch.complex128)
    for x in range(16):
        for y in range(8):
            flag = int((x >> 1) == y and (x >> 2 & 1) == 1)
            expected[16 * x + 2 * y + flag] = 1 / math.sqrt(128)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0


@pytest.mark.parametrize(
    ("operator", "literal", "predicate"),
    [
        ("==", 2, lambda value: value == 2),
        ("!=", 2, lambda value: value != 2),
        ("<", 2, lambda value: value < 2),
        ("<=", 2, lambda value: value <= 2),
        (">", 1, lambda value: value > 1),
        (">=", 2, lambda value: value >= 2),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_slice_comparison_matches_all_unsigned_values(
    operator: str, literal: int, predicate: Callable[[int], bool], reverse: bool
) -> None:
    reversed_operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(
        operator, operator
    )
    expression = (
        f"{literal} {reversed_operator} x[1:4]"
        if reverse
        else f"x[1:4] {operator} {literal}"
    )
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], flag:Output[QBit]",
            "allocate(x)\nallocate(flag)\nhadamard_transform(x)\n"
            f"control({expression}, lambda: X(flag))",
        )
    )
    actual = fq.run(program).statevector().reshape(16, 2, -1)
    expected = torch.zeros((16, 2), dtype=torch.complex128)
    for value in range(16):
        flag = int(predicate(value >> 1))
        expected[value, flag] = 0.25
    torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[:, :, 1:]) == 0


def test_out_of_range_slice_literal_retains_checked_expression_path() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], flag:Output[QBit]",
            "allocate(x)\nallocate(flag)\nhadamard_transform(x)\n"
            "control(x[1:3] == 8, lambda: X(flag))",
        )
    )
    actual = fq.run(program).statevector().reshape(16, 2, -1)
    assert torch.count_nonzero(actual[:, 1:, 0]) == 0
    assert torch.count_nonzero(actual[:, :, 1:]) == 0


def test_wide_parent_counts_only_selected_bits_and_retains_branch_phase() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[12]]",
            "allocate(x)\nhadamard_transform(x)\n"
            "control(x[9:12] == 3, lambda: phase(1, 0.37))",
        )
    )
    actual = fq.run(program).statevector().reshape(4096, -1)
    expected = torch.ones(4096, dtype=torch.complex128) / 64
    expected[1536:2048] *= complex(math.cos(0.37), math.sin(0.37))
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0


def test_slice_predicate_callback_adjoint_and_interference() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], flag:Output[QBit]",
            "allocate(x)\nallocate(flag)\nhadamard_transform(x)\nH(flag)\n"
            "kick(x, flag)\nwith adjoint():\n    kick(x, flag)\nH(flag)",
            "@qfunc\ndef kick(x: Const[QNum], flag: QBit):\n"
            "    control(x[1:4] >= 2, lambda: R(0.73, 0, flag))\n"
            "    control(x[1:4] >= 2, lambda: phase(1, 0.41))\n",
        )
    )
    actual = fq.run(program).statevector().reshape(16, 2, -1)
    torch.testing.assert_close(
        actual[:, 0, 0],
        torch.full((16,), 0.25, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24
    assert float(actual[:, :, 1:].abs().square().sum()) < 1e-24


@pytest.mark.parametrize(
    "view", ["x[-1:2]", "x[1:5]", "x[2:2]", "x[0:4:2]", "x[4]", "x[y:3]"]
)
def test_invalid_views_fail_before_lowering(view: str) -> None:
    with pytest.raises(HybridVerificationError, match="view|integer|slice"):
        synth.synthesize(
            definition(
                "x:Output[QNum[4]], y:Output[QBit], flag:Output[QBit]",
                "allocate(x)\nallocate(y)\nallocate(flag)\n"
                f"control({view} == 1, lambda: X(flag))",
            )
        )


@pytest.mark.parametrize("spec", ["4, True, 0", "4, False, 2"])
def test_signed_and_fixed_point_views_require_explicit_numeric_semantics(
    spec: str,
) -> None:
    with pytest.raises(HybridVerificationError, match="unsigned integer"):
        synth.synthesize(
            definition(
                f"x:Output[QNum[{spec}]], flag:Output[QBit]",
                "allocate(x)\nallocate(flag)\ncontrol(x[1:3] == 1, lambda: X(flag))",
            )
        )


@pytest.mark.parametrize("target", ["x[2]", "x[0]"])
def test_parent_borrowing_refuses_target_alias(target: str) -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            definition(
                "x:Output[QNum[4]]",
                f"allocate(x)\ncontrol(x[1:3] == 1, lambda: X({target}))",
            )
        )


def test_predicate_phase_survives_within_apply_interference() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QNum[4]], flag:Output[QBit]",
            "allocate(x)\nallocate(flag)\nhadamard_transform(x)\n"
            "with within():\n    control(x[1:4] == 2, lambda: X(flag))\n"
            "with apply():\n    RZ(0.61, flag)\nhadamard_transform(x)",
        )
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    diagonal = torch.full(
        (32,), complex(math.cos(-0.305), math.sin(-0.305)), dtype=torch.complex128
    )
    diagonal[8:12] = torch.full(
        (4,), complex(math.cos(0.305), math.sin(0.305)), dtype=torch.complex128
    )
    reference.any(*range(5), matrix=torch.diag(diagonal))
    for wire in range(4):
        reference.h(wire)
    actual = fq.run(program).statevector().reshape(32, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(actual[:, 1:]) == 0


def test_predicate_views_retain_actual_resource_rejection() -> None:
    function = definition(
        "x:Output[QNum[4]], flag:Output[QBit]",
        "allocate(x)\nallocate(flag)\nhadamard_transform(x)\n"
        "control(x[1:4] == 2, lambda: X(flag))",
    )
    program = synth.synthesize(function)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(function, max_operations=resources.n_operations - 1)
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(function, max_workspace=0)
    with pytest.raises(HybridVerificationError, match="eight input bits"):
        synth.synthesize(
            definition(
                "x:Output[QNum[12]], flag:Output[QBit]",
                "allocate(x)\nallocate(flag)\n"
                "control((x[0:8] == 1) | ((x[4:12] ^ x[4:12]) == 0), lambda: X(flag))",
            )
        )


def test_slice_predicate_refuses_callback_writes_and_output_escape() -> None:
    for helper, action, diagnostic in (
        ("@qfunc\ndef mutate(x: QNum):\n    x += 1\n", "mutate(x)", "disjoint"),
        (
            "@qfunc\ndef fresh(flag: Output[QBit]):\n    allocate(flag)\n",
            "fresh(flag)",
            "controlled helpers require Const-only ports",
        ),
    ):
        with pytest.raises(HybridVerificationError, match=diagnostic):
            synth.synthesize(
                definition(
                    "x:Output[QNum[4]], flag:Output[QBit]",
                    "allocate(x)\nallocate(flag)\n"
                    f"control(x[1:3] == 1, lambda: {action})",
                    helper,
                )
            )
