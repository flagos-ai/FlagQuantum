"""Verify typed QEC composite costs and caches against frozen MultiAnd."""

from collections.abc import Mapping
from typing import cast

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.program_cost_query import (
    query_program_qec_gate_cost,
)
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.multi_control import (
    multi_and_ir,
    multi_and_stages,
)
from flagquantum.compiler.registers.primitives.and_compute import and_compute_ir
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _program(
    values: tuple[int, ...],
) -> tuple[ProgramRecord, Mapping[str, ResourceCall]]:
    stages = tuple(multi_and_stages(values))
    keys: dict[str, ResourceCall] = {}
    functions = []
    bit = quantum_bit_type()
    ports = ("left", "right", "out")
    for _, _, _, controls in stages:
        name = f"and_{controls[0]}{controls[1]}"
        if name not in keys:
            keys[name] = ConjunctionCall(controls)
            functions.append(
                bit_gate_function(
                    name,
                    tuple(Register(port, bit) for port in ports),
                    CircuitIR(3, and_compute_ir(0, 1, 2, values=controls)),
                )
            )
    wires = tuple(f"wire{i}" for i in range(len(values) + len(stages)))
    builder = ComponentProgramBuilder(
        Signature(tuple(Register(name, bit) for name in wires)), functions
    )
    bindings = builder.inputs.copy()
    for left, right, target, controls in stages:
        incoming = tuple(wires[index] for index in (left, right, target))
        name = f"and_{controls[0]}{controls[1]}"
        outputs = builder.call_dict(
            name,
            {port: bindings[wire] for port, wire in zip(ports, incoming, strict=True)},
        )
        bindings.update(
            (wire, outputs[port]) for port, wire in zip(ports, incoming, strict=True)
        )
    return builder.finalize(bindings), keys


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("values", [(1, 1, 1, 1, 1), (0, 1, 1, 0, 1)])
def test_composite_qec_counts_and_cache_match_source(
    legacy: bool, values: tuple[int, ...]
) -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

    model = MultiAnd(cvs=values)
    composite = model.decompose_bloq()
    metric = source.QECGatesCost(legacy_shims=legacy)
    program, keys = _program(values)

    cache: dict[str, QECGateCost] = {}
    actual = query_program_qec_gate_cost(
        program, keys, legacy_shims=legacy, cache=cache
    )
    ref_cache = source.get_cost_cache(composite, metric)
    assert actual.asdict() == source.get_cost_value(composite, metric).asdict()
    declared = source.get_cost_value(model, metric).asdict()
    if not legacy and 0 in values:
        # Source MultiAnd's declaration explicitly charges two X per inversion.
        assert declared == {**actual.asdict(), "clifford": 2 * values.count(0)}
    else:
        assert declared == actual.asdict()
    assert {
        And(*key.controls): cache[name].asdict()
        for name, key in keys.items()
        if isinstance(key, ConjunctionCall) and isinstance(key.controls, tuple)
    } == {gate: cost.asdict() for gate, cost in ref_cache.items()}
    assert "main" not in cache
    sentinel = QECGateCost(t=99)
    cache["main"] = sentinel
    assert (
        query_program_qec_gate_cost(program, keys, legacy_shims=legacy, cache=cache)
        == actual
    )
    assert cache["main"] is sentinel
    static = QECGateCost(t=7)
    assert (
        query_program_qec_gate_cost(
            program, {}, cache=cache, static_costs={"main": static}
        )
        is static
    )
    assert cache["main"] is static

    with pytest.raises(TypeError, match="Static gate counts require"):
        query_program_qec_gate_cost(
            program, keys, static_costs={"main": cast(QECGateCost, 7)}
        )
    with pytest.raises(ValueError, match="Missing QEC terminal key"):
        query_program_qec_gate_cost(program, {})


def test_appended_five_control_ladder_has_separate_emitted_counts() -> None:
    values = (1, 1, 1, 1, 1)
    program, keys = _program(values)
    physical = bind_register_program(program, tuple((i,) for i in range(9))).circuit
    expected = multi_and_ir((0, 1, 2, 3, 4), values, (5, 6, 7, 8))
    assert physical.instructions == expected
    model = query_program_qec_gate_cost(program, keys)
    assert model == QECGateCost(and_bloq=4)
    assert sum(op.name in {"t", "tdg"} for op in physical.instructions) == 16
    state = torch.zeros(512, dtype=torch.complex128)
    state[(30 << 4)] = torch.tensor(0.5j, dtype=torch.complex128)
    state[(31 << 4)] = 3**0.5 / 2
    output = torch.zeros_like(state)
    output[(30 << 4) + 14] = state[30 << 4]
    output[(31 << 4) + 15] = state[31 << 4]
    circuit = Circuit(9, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(9, physical.instructions))
    torch.testing.assert_close(circuit.state().flatten(), output, rtol=0, atol=2e-12)
