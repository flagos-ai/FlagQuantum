"""Compare specialized source control graphs, including their negative-bit defect."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.indexed_component_control import (
    specialized_indexed_component_ir,
)
from flagquantum.compiler.registers.indexed_component_control_resource_model import (
    specialized_indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit


def _native(
    values: tuple[int, ...], existing: int | None, adjoint: bool, grouped: bool = False
):
    count = len(values)
    selection = count + int(existing is not None)
    target = selection + 1
    data = target + 1
    work = tuple(range(data, data + count + int(existing is not None)))
    bits = tuple(range(40, 40 + len(work)))
    gates = specialized_indexed_component_ir(
        ((selection,),),
        (2,),
        lambda index, predicate: (
            Instruction("cx" if index == 0 else "cz", (predicate, target)),
        ),
        (tuple(range(count)),),
        (
            (
                ControlRegister(
                    len(values),
                    (
                        sum(
                            bit << (len(values) - i - 1) for i, bit in enumerate(values)
                        ),
                    ),
                )
                if grouped
                else ControlRegister(1, values)
            ),
        ),
        current_control=None if existing is None else (count, existing),
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    return gates, work, bits, selection, target


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
@pytest.mark.parametrize("existing", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_original_control_graphs_and_resources(values, existing, adjoint, grouped):
    import numpy as np
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    base = ApplyLthBloq(np.array([XGate(), ZGate()]), control_val=existing)
    source = (base.adjoint() if adjoint else base).controlled(
        CtrlSpec(
            qdtypes=QUInt(len(values)),
            cvs=sum(bit << (len(values) - i - 1) for i, bit in enumerate(values)),
        )
        if grouped
        else CtrlSpec(cvs=values[0] if len(values) == 1 else values)
    )
    gates, work, bits, selection, target = _native(values, existing, adjoint, grouped)
    ports = {"selection": (selection,), "q": (target,)}
    if len(values) == 1:
        ports["control"] = tuple(range(selection))
    else:
        ports["ctrl"] = tuple(range(len(values)))
        if existing is not None:
            ports["control"] = (len(values),)
    expected = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement
    report = estimate_ftqc_resources(
        specialized_indexed_component_cost(
            (1,),
            (2,),
            (
                ComponentCost("CX", 2, GateCost(clifford=1)),
                ComponentCost("CZ", 2, GateCost(clifford=1)),
            ),
            (ControlRegister(1, values),),
            target_width=1,
            current_control_value=existing,
            adjoint=adjoint,
        )
    )
    assert asdict(report.logical.gates) == {
        name: getattr(model, name) for name in asdict(report.logical.gates)
    }
    assert report.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert get_cost_value(source, QubitCount()) == target + 1 + len(work)


@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
@pytest.mark.parametrize("existing", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_every_complex_branch_and_clean_owned_work(values, existing, adjoint):
    gates, work, _, selection, target = _native(values, existing, adjoint)
    data = target + 1
    total = data + len(work)
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word in range(1 << data):
        amp = complex(word + 1, 2 - word) / (1 << data)
        initial[word << len(work)] = amp
        control_bits = tuple(
            (word >> (data - wire - 1)) & 1 for wire in range(len(values))
        )
        # Literal locked source specialization ignores a solitary activation zero.
        active = control_bits == (
            (1,) if len(values) == 1 and existing is None else values
        )
        if existing is not None:
            active &= ((word >> (data - len(values) - 1)) & 1) == existing
        address = (word >> (data - selection - 1)) & 1
        output = word ^ int(active and address == 0)
        sign = -1 if active and address == 1 and word & 1 else 1
        expected[output << len(work)] = sign * amp
    measurements = sum(g.name == "measure" for g in gates)
    for branch in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, gates, branch),
            expected * 2 ** (-measurements / 2),
            atol=3e-12,
            rtol=0,
        )


def test_runnable_append_and_estimation_example(capsys):
    from examples.algorithms.indexed_component_controls import main

    main()
    output = capsys.readouterr().out
    assert "Model T equivalents 12" in output
    assert "Emitted T/T-dagger instructions 12" in output
    assert "Model logical peak 8" in output
    assert "Modern model Clifford count 6" in output
    assert "Legacy model Clifford count 22" in output


@pytest.mark.parametrize(
    "change",
    [
        {"controls": ()},
        {"workspace": ()},
        {"classical_bits": (0, 0)},
        {"current_control": (0, 1)},
        {"current_control": (4, 2)},
        {"adjoint": 1},
    ],
)
def test_invalid_bindings_fail_before_child_emission(change):
    def refuse(index, predicate):
        pytest.fail("Invalid control layout reached the child")

    args = dict(
        axes=((1,),),
        extents=(2,),
        emit_controlled=refuse,
        controls=((0,),),
        registers=(ControlRegister(1, (1,)),),
        workspace=(2,),
        classical_bits=(0,),
    )
    args.update(change)
    with pytest.raises(ValueError):
        specialized_indexed_component_ir(**args)
