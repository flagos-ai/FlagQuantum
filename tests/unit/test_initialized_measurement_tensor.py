"""Zero-input phaseful measurement/discard through existing native owners."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import discard_word_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.component_zero_initialization import (
    initialize_component_from_zero,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
)
from flagquantum.simulation.tensor_network.register_zero_state import (
    register_zero_state_nodes,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_superoperator_nodes import supplier
from tests.unit.test_component_zero_initialization import phase_fixture
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def fixture(basis: str, discard: bool) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Discard, Hadamard, MeasureX, MeasureZ, SGate

    phase, _ = phase_fixture(1, ())
    h, s = phase.functions[1:3]
    measurement = (
        measured_function("first")
        if basis == "z"
        else measure_x_word_function("second")
    )
    registers = [
        Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
        Register("idle", quantum_bit_type()),
    ]
    if not discard:
        registers.append(Register("c", classical_bit_type(), side=RegisterSide.RIGHT))
    builder = ComponentProgramBuilder(
        Signature(registers), (h, s, measurement, discard_word_function("drop"))
    )
    q = builder.call_dict("h", {"q": builder.inputs["q"]})["q"]
    q = builder.call_dict("s", {"q": q})["q"]
    c = builder.call_dict(measurement.name, {"q": q})["c"]
    outputs = {"idle": builder.inputs["idle"]}
    if discard:
        builder.call_dict("drop", {"c": c})
    else:
        outputs["c"] = c
    bb, vals = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister("q", QBit(), side=Side.LEFT),
                SourceRegister("idle", QBit()),
                *([] if discard else [SourceRegister("c", CBit(), side=Side.RIGHT)]),
            ]
        )
    )
    sq = bb.add(SGate(), q=bb.add(Hadamard(), q=vals["q"]))
    sc = bb.add(MeasureZ() if basis == "z" else MeasureX(), q=sq)
    src_outputs = {"idle": vals["idle"]}
    if discard:
        bb.add(Discard(), c=sc)
    else:
        src_outputs["c"] = sc
    return builder.finalize(outputs), bb.finalize(**src_outputs)


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("atol", [None, 1e-12])
def test_initialized_phase_measurement_outputs_and_discard(
    basis: str, discard: bool, atol: float | None
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor import initialize_from_zero

    record, source = fixture(basis, discard)
    initialized = initialize_component_from_zero(record)
    reference = initialize_from_zero(source)
    like = torch.empty((), dtype=torch.complex128)
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    numerical = supplier(like)

    def supply(
        function: FunctionRecord,
        incoming: Ports,
        outgoing: Ports,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        assert function.registers is not None
        signature = Signature(function.registers)
        if function.name.startswith("__fq_zero_"):
            return register_zero_state_nodes(signature, outgoing, like)
        if function.name in {"first", "second", "drop"}:
            return numerical(function, incoming, outgoing, fresh)
        matrix = (
            torch.tensor([[1, 1], [1, -1]], dtype=like.dtype) / 2**0.5
            if function.name == "h"
            else torch.diag(
                torch.stack((phase * 0 + 1, torch.exp(1j * (phase + torch.pi / 2))))
            )
        )
        return (bind_register_tensor(matrix, signature, incoming, outgoing),)

    names = [
        f.name
        for f in initialized.functions
        if f.name != "from_zero" and f.name != "main"
    ]
    providers = ComponentTensorProviders(dict.fromkeys(names, supply))
    selected, _ = expand_components_for_tensor(
        initialized, providers.has_custom_tensor, "from_zero", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "from_zero"))
    # The unmodified upstream represents the exact S phase, before the native
    # trainable perturbation. Compare at phase0 and check nonzero-phase analytically.
    phase_zero = phase.detach().clone()
    with torch.no_grad():
        phase.zero_()
    source_point = contract_component_superoperator(
        selected, order, providers.supply, like, "from_zero"
    )
    np.testing.assert_allclose(
        source_point.detach().numpy(),
        reference.tensor_contract(superoperator=True),
        rtol=0,
        atol=4e-16,
    )
    with torch.no_grad():
        phase.copy_(phase_zero)
    actual = contract_component_superoperator(
        selected, order, providers.supply, like, "from_zero", simplification_atol=atol
    )
    psi = torch.stack((phase * 0 + 1, torch.exp(1j * (phase + torch.pi / 2)))) / 2**0.5
    if basis == "x":
        psi = torch.tensor([[1, 1], [1, -1]], dtype=like.dtype) @ psi / 2**0.5
    probabilities = psi.abs().square()
    expected = torch.zeros_like(actual)
    if discard:
        expected[0, 0] = probabilities.sum()
    else:
        expected.reshape(2, 2, 2, 2)[0, :, 0, :] = torch.diag(probabilities).to(
            like.dtype
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-16)
    # A bounded observable avoids scaling roundoff with the dense axis count.
    weights = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape)
    weights = weights / max(1, actual.numel() - 1)
    torch.testing.assert_close(
        torch.autograd.grad((actual.real * weights).sum(), phase, retain_graph=True)[0],
        torch.autograd.grad((expected.real * weights).sum(), phase)[0],
        rtol=0,
        atol=5e-16,
    )
    # Numerical initialization does not grant physical retirement ownership for
    # a pool-allocated word. Keep that existing refusal; bind the original graph
    # with caller-owned zero inputs for the physical branch comparison.
    with pytest.raises(ValueError, match="retirement ownership contract"):
        bind_register_program(
            initialized,
            (),
            entry="from_zero",
            clean=(0, 1, 2),
            allow_workspace_outputs=True,
        )
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    gates = bound.circuit.instructions
    counts = Counter(g.name for g in gates)
    assert counts == {"h": 2 if basis == "x" else 1, "s": 1, "measure": 1}
    assert all(g.wires == (0,) and not g.params for g in gates)
    assert gates[-1].metadata["classical_bit"] == 0
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit

    assert Circuit(3).append(FieldOperation(3, gates)).to_ir().instructions == gates
    initial = torch.zeros(8, dtype=like.dtype)
    initial[0] = 1
    # Each physical branch retains its exact complex phase and unused work is0.
    for bit in (0, 1):
        branch = forced_branch(initial, gates, (bit,))
        assert torch.count_nonzero(branch.reshape(-1, 2)[:, 1:]) == 0
        nonzero = branch[branch != 0]
        assert nonzero.numel() == 1
        exact = (
            (1 if bit == 0 else 1j) / 2**0.5
            if basis == "z"
            else (1 + (1j if bit == 0 else -1j)) / 2
        )
        torch.testing.assert_close(
            nonzero[0], torch.tensor(exact, dtype=like.dtype), rtol=0, atol=4e-16
        )
    model = get_cost_value(reference, QECGatesCost())
    assert model.measurement == 1 and model.clifford == 4
    assert model.total_t_count() == 0
