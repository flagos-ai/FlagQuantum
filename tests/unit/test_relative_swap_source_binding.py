"""Resolve the original relative-swap factory through an owned IR roundtrip."""

from collections import Counter
from dataclasses import replace
from math import sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.swap_network_catalog import RelativeSwapExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_swap_network_file import source_network as source_network

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("activation", [None, 0, 1, (0, 1)])
def test_bound_original_relative_swap_complete_operators(
    source_network: Any, width: int, activation: int | tuple[int, ...] | None
) -> None:
    from attrs import evolve
    from qualtran import CtrlSpec
    from qualtran.bloqs.swap_network.cswap_approx import _approx_cswap_symb

    source = evolve(_approx_cswap_symb.make(), bitsize=width)
    if activation is not None:
        source = source.controlled(CtrlSpec(cvs=activation))
    original = import_register_emitter(source)
    emitter = replace(
        original,
        snapshots=(
            CircuitIR.from_json(original.snapshots[0].to_json()),
            CircuitIR.from_json(original.snapshots[1].to_json()),
        ),
    )
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    logical = sum(bits for _, bits in ports)
    work, slots = emitter.requirements()
    binding = BlackBoxPrepare(ports, (), emitter)
    operators = []
    for adjoint in (False, True):
        native = RelativeSwapExample(width, activation, adjoint)
        expected = Circuit(native.qubit_count).append(native).to_ir()
        captured = (
            Circuit(logical + work)
            .append(
                binding.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert isinstance(captured, CircuitIR) and isinstance(expected, CircuitIR)
        assert Counter(g.name for g in captured.instructions) == Counter(
            g.name for g in expected.instructions
        )
        matrices = []
        for ir in (captured, expected):
            count = certify_erasure_histories(
                ir.instructions,
                n_wires=ir.n_wires,
                zero_wires=tuple(range(logical, ir.n_wires)),
                check_clean_output=False,
            )
            stride = 1 << (ir.n_wires - logical)
            columns = []
            for column in range(1 << logical):
                initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
                initial[column * stride] = 1
                result = forced_branch(initial, ir.instructions, (0,) * count) * sqrt(
                    2**count
                )
                clean = torch.zeros_like(result)
                clean[::stride] = result[::stride]
                torch.testing.assert_close(result, clean, atol=8e-12, rtol=0)
                columns.append(result[::stride])
            matrices.append(torch.stack(columns, dim=1))
        torch.testing.assert_close(matrices[0], matrices[1], atol=8e-12, rtol=0)
        operators.append(matrices[0])
    forward, backward = operators
    torch.testing.assert_close(
        backward @ forward,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    # Approximate exchange means the original relative phases are observable;
    # it does not authorize comparison of basis permutations alone.
    if activation is None:
        assert torch.any(forward.real < -0.5)
