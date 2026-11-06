"""Compare complete sparse chemistry preparation with the pinned source."""

from cmath import exp
from collections import defaultdict
from itertools import product
from math import sqrt

import numpy as np
import pytest

from flagquantum.algorithms.primitives.sparse_chemistry_data import (
    sparse_chemistry_data,
)
from flagquantum.algorithms.primitives.sparse_chemistry_prepare import (
    PrepareSparseChemistry,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.register_source_graph import walk_source_graph
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = pytest.mark.unit


def _branch(gates, width, history, state=None):
    """Evaluate the small-support fixture without allocating dense bank states."""
    state = {0: 1 + 0j} if state is None else state
    bits = {}
    outcomes = iter(history)
    phase = {
        "t": (1 + 1j) / sqrt(2),
        "tdg": (1 - 1j) / sqrt(2),
        "s": 1j,
        "sdg": -1j,
        "z": -1,
    }
    for gate in gates:
        if any(
            bits[bit] != value for bit, value in gate.metadata.get("conditions", ())
        ):
            continue
        masks = tuple(1 << (width - 1 - wire) for wire in gate.wires)
        mask = masks[-1]
        if gate.name == "measure":
            outcome = next(outcomes)
            bits[gate.metadata["classical_bit"]] = outcome
            state = {
                index: amp
                for index, amp in state.items()
                if bool(index & mask) == outcome
            }
            continue
        updated = defaultdict(complex)
        for index, amp in state.items():
            if gate.name == "h":
                updated[index & ~mask] += amp / sqrt(2)
                updated[index | mask] += (-amp if index & mask else amp) / sqrt(2)
            elif gate.name == "x" or (gate.name == "cx" and index & masks[0]):
                updated[index ^ mask] += amp
            elif gate.name == "reset":
                updated[index & ~mask] += amp
            elif gate.name == "rz":
                theta = gate.params["theta"]
                updated[index] += amp * exp((1j if index & mask else -1j) * theta / 2)
            elif gate.name == "cz":
                updated[index] += -amp if all(index & m for m in masks) else amp
            else:
                assert gate.name in (*phase, "cx")
                updated[index] += amp * phase.get(gate.name, 1) if index & mask else amp
        state = {index: amp for index, amp in updated.items() if abs(amp) > 1e-12}
    assert next(outcomes, None) is None
    return state


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("spatial,log", [(2, 0), (2, 1), (3, 0), (3, 1)])
def test_prepare_forward_source_graph_and_resources(spatial, log):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic import LessThanEqual
    from qualtran.bloqs.basic_gates import CSwap, Hadamard, OnEach
    from qualtran.bloqs.basic_gates.z_basis import CZ
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse
    from qualtran.bloqs.data_loading.qroam_clean import (
        QROAMClean,
        QROAMCleanAdjointWrapper,
    )
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    one = np.eye(spatial)
    one[1, 0] = one[0, 1] = -0.75
    two = np.zeros((spatial,) * 4)
    # Four or eight total rows. Keep the uniform stage power-of-two.
    two[0, 0, 0, 0] = -0.5
    if spatial == 3:
        two[2, 2, 2, 2] = 0.25
    data = sparse_chemistry_data(one, two, probability_bits=1, log_block_size=log)
    operation = PrepareSparseChemistry(data)
    reference = PrepareSparse.from_hamiltonian_coeffs(
        2 * spatial, one, two, num_bits_state_prep=1, log_block_size=log
    )
    layout = (*operation.selection_layout, *operation.junk_layout)
    assert layout == tuple((reg.name, reg.total_bits()) for reg in reference.signature)
    width = sum(bits for _, bits in layout)
    ports = partition_register(tuple(range(width)), layout)
    work, slots = operation.requirements()
    scratch = tuple(range(width, width + work))
    bits = tuple(range(slots))
    circuit = DynamicCircuit(width + work).append(
        operation.make_on(ports, workspace=scratch, classical_bits=bits)
    )
    actual = circuit.to_ir().instructions
    inverse = (
        DynamicCircuit(width + work)
        .append(
            operation.make_coherent_adjoint_on(
                ports, workspace=scratch, classical_bits=bits
            )
        )
        .to_ir()
        .instructions
    )
    source_adjoint = reference.adjoint()
    assert_source_model(operation.resource_model(adjoint=True), source_adjoint)
    lookup_work = data.lookup_table().workspace_size(0, (data.address_width,))

    def leaf(bloq, inputs, desired):
        child = getattr(bloq, "subbloq", bloq)
        if isinstance(child, PrepareSparse):
            return None

        if isinstance(bloq, QROAMClean):
            desired = dict(inputs)
            for i, (name, _) in enumerate(data.target_layout):
                desired[f"target{i}_"] = ports[name]
                if log:
                    desired[f"junk_target{i}_"] = ports[f"jnk_{name}"]
            gates, _, _ = source_clean_graph(
                bloq, inputs, desired, scratch[:lookup_work], bits[:lookup_work]
            )
            return gates, desired
        if isinstance(bloq, QROAMCleanAdjointWrapper):
            from qualtran import DecomposeNotImplementedError

            with pytest.raises(DecomposeNotImplementedError):
                bloq.qroam_clean_adjoint_bloq.decompose_bloq()
            address = {"selection": inputs["selection"]}
            forward, load_size, exchanges = source_clean_graph(
                bloq.qroam_clean,
                address,
                inputs,
                scratch[:lookup_work],
                bits[:lookup_work],
            )
            return (
                *(
                    gate
                    for exchange in reversed(exchanges)
                    for gate in source_relative_swap(
                        exchange["ctrl"][0], exchange["x"], exchange["y"]
                    )
                ),
                *forward[:load_size],
            ), address
        if isinstance(child, (PrepareUniformSuperposition, Hadamard, OnEach)):
            gates = tuple(
                Instruction("h", (wire,)) for word in inputs.values() for wire in word
            )
        elif isinstance(bloq, LessThanEqual):
            native_measurements = {
                gate.wires[0]: gate.metadata["classical_bit"]
                for gate in actual
                if gate.name == "measure"
            }
            gates = reference_arithmetic_gates(
                bloq, inputs, scratch[:2], native_measurements, cirq_style=True
            )
        elif isinstance(bloq, CSwap):
            gates = reference_arithmetic_gates(
                bloq, inputs, (), {}, preserve_components=True
            )
        elif isinstance(bloq, CZ):
            gates = (Instruction("cz", (*inputs["q1"], *inputs["q2"])),)
        else:
            assert str(bloq) == "C[ZGate]"
            assert tuple(bloq.ctrl_spec.cvs[0].flat) == (0,)
            control = inputs["ctrl"]
            flip = Instruction("x", control)
            gates = (flip, Instruction("cz", (*control, *inputs["q"])), flip)
        return gates, inputs

    expected, returned = walk_source_graph(
        reference,
        {reg.name: ports[reg.name] for reg in reference.signature.lefts()},
        ports,
        leaf,
    )
    assert returned == ports
    assert_source(actual, expected)
    source_inverse, inverse_outputs = walk_source_graph(
        source_adjoint,
        ports,
        {reg.name: ports[reg.name] for reg in source_adjoint.signature.rights()},
        leaf,
    )
    assert set(inverse_outputs) == {reg.name for reg in reference.signature.lefts()}
    assert_source(inverse, source_inverse)
    model = operation.resource_model()
    assert_source_model(model, reference)
    if spatial == 2:
        events = sum(gate.name == "measure" for gate in actual)
        if log == 0:
            assert events == 3
            assert sum(gate.name in {"t", "tdg"} for gate in actual) == 75
        assert model.gates.cswap == 8
        inverse_events = sum(gate.name == "measure" for gate in inverse)
        assert inverse_events == events
        for history in product((0, 1), repeat=events):
            output = _branch(actual, width + work, history)
            assert output == pytest.approx(
                _branch(expected, width + work, history), abs=2e-12
            )
            assert all(index & ((1 << work) - 1) == 0 for index in output)
            assert len(output) == 64
            assert sum(abs(amp) ** 2 for amp in output.values()) == pytest.approx(
                2**-events
            )
            for inverse_history in product((0, 1), repeat=inverse_events):
                restored = _branch(inverse, width + work, inverse_history, output)
                assert restored == pytest.approx({0: 2**-events}, abs=2e-12)
            if log == 0:
                assert output == pytest.approx(
                    _expected_unblocked(data, layout, work, events), abs=2e-12
                )


def _expected_unblocked(data, layout, work, events):
    """Compute the signed alias selection and symmetry action classically."""
    result = {}
    for row, sigma, pq, rs, pqrs in product(range(4), *(range(2) for _ in range(4))):
        values = dict(
            zip(
                (name for name, _ in data.target_layout),
                (column[row] for column in data.datasets),
                strict=True,
            )
        )
        flag = int(values["keep"] <= sigma)
        sign = values["alt_theta"] if flag else values["theta"]
        if flag:
            for name in ("p", "q", "r", "s", "flag_1b"):
                alt = f"alt_{name}"
                values[name], values[alt] = values[alt], values[name]
        for enabled, left, right in (
            (pqrs, "p", "r"),
            (pqrs, "q", "s"),
            (pq, "p", "q"),
            (rs, "r", "s"),
        ):
            if enabled:
                values[left], values[right] = values[right], values[left]
        values.update(
            d=row, sigma=sigma, swap_pq=pq, swap_rs=rs, swap_pqrs=pqrs, less_than=flag
        )
        index = 0
        for name, bits in layout:
            index = (index << bits) | values.get(name, 0)
        result[index << work] = (-1) ** sign / sqrt(64 * 2**events)
    return result


@pytest.mark.ftqc_reference
def test_padded_lookup_adjoint_and_non_power_of_two_uniform_round_trip():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse

    one = np.array([[1.0, -0.75], [-0.75, 1.0]])
    two = np.zeros((2,) * 4)
    data = sparse_chemistry_data(one, two, probability_bits=1, log_block_size=1)
    prepare = PrepareSparseChemistry(data)
    reference = PrepareSparse.from_hamiltonian_coeffs(
        4, one, two, num_bits_state_prep=1, log_block_size=1
    )
    assert len(data.indices) == 3
    assert_source_model(prepare.resource_model(adjoint=True), reference.adjoint())
    layout = (*prepare.selection_layout, *prepare.junk_layout)
    width = sum(bits for _, bits in layout)
    ports = partition_register(tuple(range(width)), layout)
    work, slots = prepare.requirements()
    scratch = tuple(range(width, width + work))
    bits = tuple(range(slots))
    forward = prepare.make_on(
        ports, workspace=scratch, classical_bits=bits
    ).instructions
    inverse = prepare.make_coherent_adjoint_on(
        ports, workspace=scratch, classical_bits=bits
    ).instructions
    events = sum(gate.name == "measure" for gate in forward)
    assert events == sum(gate.name == "measure" for gate in inverse)
    for history in product((0, 1), repeat=events):
        output = _branch(forward, width + work, history)
        assert output and all(index & ((1 << work) - 1) == 0 for index in output)
        restored = _branch(
            inverse, width + work, tuple(1 - bit for bit in history), output
        )
        assert restored == pytest.approx({0: 2**-events}, abs=2e-12)
