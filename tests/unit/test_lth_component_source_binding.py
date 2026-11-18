"""Exercise complete indexed Bloq bindings with owned directional snapshots."""

from collections import Counter
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_lth_component_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("shape", ((2,), (2, 2)))
@pytest.mark.parametrize("control", (None, 0, 1))
def test_complete_source_binding_preserves_directions_and_control_behavior(
    source: Any, shape: tuple[int, ...], control: int | None
) -> None:
    import numpy as np
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate

    count = prod(shape)
    leaves = (TGate(), Hadamard(), ZGate(), XGate())[:count]
    original = source.ApplyLthBloq(
        np.asarray(leaves, dtype=object).reshape(shape), control_val=control
    )
    emitter = import_register_emitter(original)
    selection, target = emitter.layout[:-1], emitter.layout[-1:]
    operation = SelectBlockEncoding(
        selection, target, emitter, BlackBoxPrepare(selection, (), _identity_prepare)
    )
    controls = int(control is not None)
    logical = controls + len(shape) + 1
    work, bits = emitter.requirements()
    costs = (
        GateCost(rotation=1),
        GateCost(clifford=4, rotation=2),
        GateCost(clifford=1),
        GateCost(clifford=1),
    )
    children = tuple(
        ComponentCost(str(leaf), 2, cost)
        for leaf, cost in zip(leaves, costs[:count], strict=True)
    )
    identity = torch.eye(2, dtype=torch.complex128)
    matrices = tuple(
        torch.as_tensor(leaf.tensor_contract(), dtype=torch.complex128)
        for leaf in leaves
    )

    for adjoint in (False, True):
        ir = (
            Circuit(logical + work)
            .append(
                operation.make_on(
                    selection=tuple(range(logical - 1)),
                    target=(logical - 1,),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(bits)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        assert CircuitIR.from_json(ir.to_json()).instructions == ir.instructions
        inventory = Counter(gate.name for gate in ir.instructions)
        events = inventory["measure"]
        assert events == count + controls - 2
        assert inventory["t"] + inventory["tdg"] == 4 * events
        assert inventory["cphase"] == inventory["ch"] == 1
        model = indexed_component_cost(
            (1,) * len(shape),
            shape,
            children,
            target_width=1,
            control_count=controls,
            adjoint=adjoint,
        )
        assert_source_model(model, original.adjoint() if adjoint else original)
        # The locked direct control_val=0 path also activates on one.
        for enabled, address, bit in product(
            range(1 << controls), range(count), (0, 1)
        ):
            matrix = matrices[address] if not controls or enabled else identity
            if adjoint:
                matrix = matrix.mH
            prefix = (enabled * count + address) * 2
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            expected = torch.zeros_like(initial)
            initial[(prefix + bit) << work] = 1
            expected[prefix << work : (prefix + 2) << work : 1 << work] = matrix[:, bit]
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial, ir.instructions, history),
                    expected / 2 ** (events / 2),
                    atol=3e-12,
                    rtol=0,
                )
