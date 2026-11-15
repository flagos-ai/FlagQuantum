"""Reconcile the original four-term signed block and full complex execution."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.linear_combination_catalog import (
    linear_combination_example,
    linear_combination_example_circuit,
    linear_combination_example_model,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_four_term_signature_signal_and_cost() -> None:
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source = _linear_combination_block_encoding()
    encoding = linear_combination_example()
    widths = (
        encoding.system_bitsize,
        encoding.ancilla_bitsize,
        encoding.resource_bitsize,
    )
    assert widths == tuple(int(register.total_bits()) for register in source.signature)
    assert (encoding.alpha, encoding.epsilon) == (source.alpha, source.epsilon)
    assert encoding.coefficients == tuple(source._lambd)
    assert encoding.signal_state.selection_width == 2
    assert encoding.signal_state.emit((1, 2), (), (), ()) == ()
    model = linear_combination_example_model()
    source_calls = source.call_graph(max_depth=1, generalizer=ignore_split_join)[1]
    assert tuple(call.repetitions for call in model.calls) == tuple(
        source_calls.values()
    )
    resource = estimate_component_resources(model)
    assert {
        key: value for key, value in asdict(resource.gates).items() if value
    } == get_cost_value(source, QECGatesCost()).asdict()
    assert resource.peak_logical_qubits == get_cost_value(source, QubitCount())
    for adjoint in (False, True):
        ir = linear_combination_example_circuit(adjoint=adjoint).to_ir()
        emitted = count_emitted_instructions(ir)
        assert emitted.t_count == 4 * resource.gates.and_bloq + 7 * resource.gates.cswap
        assert emitted.unconditional["measure"] == resource.gates.measurement
        target = [gate.name for gate in ir.instructions if gate.wires[-1:] == (0,)]
        assert Counter(target) == {"cphase": 1, "ch": 1, "cx": 1, "cz": 1}


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_four_term_source_gate_operands_and_feedback(adjoint: bool) -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    encoding = linear_combination_example()
    source = _linear_combination_block_encoding()
    if adjoint:
        source = source.adjoint()
    circuit = linear_combination_example_circuit(adjoint=adjoint)
    width = (
        encoding.system_bitsize + encoding.ancilla_bitsize + encoding.resource_bitsize
    )
    ports = {"system": (0,), "ancilla": (1, 2), "resource": tuple(range(3, width))}
    work = tuple(range(width, circuit.n_qubits))
    prep_work, prep_slots = encoding.coefficient_preparation.workspace_requirements
    select_work = indexed_component_workspace((encoding.select.selection_bitsize,), 0)

    def leaf(
        bloq: Any,
        inputs: SourcePorts,
        desired: SourcePorts,
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, BlackBoxPrepare):
            declared = encoding.prepare.emit(
                inputs["selection"],
                inputs.get("junk", ()),
                work[:prep_work],
                tuple(range(prep_slots)),
                adjoint=isinstance(bloq, Adjoint),
            )
            slots = {
                gate.wires[0]: gate.metadata["classical_bit"]
                for gate in declared
                if gate.name == "measure"
            }
            used_work = work[:prep_work]
        elif isinstance(base, BlackBoxSelect):
            slots = dict(zip(work[:select_work], range(select_work), strict=True))
            used_work = work[:select_work]
        else:
            return None
        gates = reference_arithmetic_gates(
            bloq.decompose_bloq(),
            inputs,
            used_work,
            slots,
            cirq_style=True,
            preserve_components=True,
            keep_pauli_leaves=True,
            keep_exchange_leaves=True,
        )
        return gates, inputs

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_dependencies(circuit.to_ir().instructions, expected)


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_four_term_full_complex_amplitudes_and_work(adjoint: bool) -> None:
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )

    source = _linear_combination_block_encoding()
    if adjoint:
        source = source.adjoint()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    circuit = linear_combination_example_circuit(adjoint=adjoint)
    gates = circuit.to_ir().instructions
    logical = matrix.shape[0].bit_length() - 1
    reserved = circuit.n_qubits - logical
    inputs = torch.arange(matrix.shape[0], dtype=torch.float64)
    data = torch.complex(inputs + 1, inputs % 13 - 5)
    data /= data.norm()
    zeros = torch.zeros(1 << reserved, dtype=torch.complex128)
    zeros[0] = 1
    initial = torch.kron(data, zeros)
    expected = torch.kron(matrix @ data, zeros)
    measurements = sum(gate.name == "measure" for gate in gates)
    assert measurements == 8
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**measurements),
            rtol=0,
            atol=4e-12,
        )
