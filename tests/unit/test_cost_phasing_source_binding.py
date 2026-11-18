"""Bind actual external RIGHT cost oracles to native compute-phase-erase stages."""

from collections import Counter
from dataclasses import replace
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.phase.cost_function_phasing import (
    RegisterPorts,
    cost_function_phasing_ir,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    qvr_phase_gradient_ir,
    source_qvr_phase_gradient_cost,
)
from flagquantum.compiler.registers.phase.qvr_zpow import (
    qvr_zpow_ir,
    source_qvr_zpow_cost,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_cost_phasing_file import source_module as source_module
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_phase_gradient_source_binding import through_operators

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("coefficient", [0.375, -0.375])
def test_external_symbolic_cost_oracle_complete_replacement(
    source_module: Any, gradient: bool, coefficient: float
) -> None:
    import attrs
    from qualtran import (
        BloqBuilder,
        GateWithRegisters,
        QAny,
        QFxp,
        Register,
        Side,
        Signature,
    )
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )
    from sympy import symbols

    def oracle_signature(self: Any) -> Any:
        return Signature(
            (
                Register("data", QAny(self.bitsize)),
                Register("cost", QFxp(self.bitsize, self.bitsize), side=Side.RIGHT),
            )
        )

    def oracle_build(self: Any, bb: BloqBuilder, **soqs: Any) -> Any:
        data = bb.split(soqs["data"])
        cost = bb.split(bb.allocate(dtype=QFxp(self.bitsize, self.bitsize)))
        for i in range(self.bitsize):
            data[i], cost[i] = bb.add(CNOT(), ctrl=data[i], target=cost[i])
        return {
            "data": bb.join(data, dtype=QAny(self.bitsize)),
            "cost": bb.join(cost, dtype=QFxp(self.bitsize, self.bitsize)),
        }

    # Construct the optional SDK fixture at its untyped boundary. Native
    # callbacks below use the existing typed RegisterPorts interface.
    copy_cost_type = attrs.frozen(
        type(
            "CopyCost",
            (GateWithRegisters,),
            {
                "__module__": __name__,
                "__annotations__": {"bitsize": object},
                "signature": property(oracle_signature),
                "build_composite_bloq": oracle_build,
            },
        )
    )

    n, g, e = symbols("n gamma epsilon", positive=True)
    phase_type = QvrPhaseGradient if gradient else QvrZPow
    phase = phase_type(Register("cost", QFxp(n, n)), g, e)
    declaration = source_module.PhasingViaCostFunction(copy_cost_type(n), phase)
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    phase = attrs.evolve(
        phase, cost_reg=Register("cost", QFxp(2, 2)), gamma=coefficient, eps=0.2
    )
    source = attrs.evolve(
        declaration,
        cost_eval_oracle=attrs.evolve(declaration.cost_eval_oracle, bitsize=2),
        phase_oracle=phase,
    )
    b = int(phase.b_grad) if gradient else 0
    logical = 2 + b
    fmt = qvr_phase_format(2, 2, coefficient, 0.2) if gradient else None
    terms = (
        scaled_phase_add_terms(
            2, 2, b, coefficient, fmt.coefficient_width, fmt.coefficient_fraction_bits
        )
        if fmt
        else ()
    )
    phase_work = max(
        (
            phase_add_workspace_size(len(tuple(range(2))[part]), b, right_shift=shift)
            for part, shift in terms
        ),
        default=0,
    )
    cost = tuple(range(logical, logical + 2))
    work = tuple(range(logical + 2, logical + 2 + phase_work))
    data = (0, 1)
    grad = tuple(range(2, logical))

    def evaluate(ports: RegisterPorts) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (x, y))
            for x, y in zip(ports["data"], ports["cost"], strict=True)
        )

    def emit_phase(
        ports: RegisterPorts, inverse: bool = False
    ) -> tuple[Instruction, ...]:
        if gradient:
            return qvr_phase_gradient_ir(
                ports["cost"],
                ports["phase_grad"],
                ports["phase_work"],
                tuple(range(phase_work)),
                cost_fraction_bits=2,
                coefficient=coefficient,
                epsilon=0.2,
                adjoint=inverse,
            )
        return qvr_zpow_ir(
            ports["cost"],
            cost_fraction_bits=2,
            coefficient=coefficient,
            epsilon=0.2,
            adjoint=inverse,
        )

    # Scratch is explicit in the private phase-stage binding, and is absent from
    # the public logical signature. Evaluation cannot borrow the phase work.
    native = cost_function_phasing_ir(
        {"data": data},
        {"cost": cost},
        {"phase_grad": grad, "phase_work": work} if gradient else {},
        cost_names=("cost",),
        evaluate=evaluate,
        phase=emit_phase,
        erase=evaluate,
    )
    native_ir = CircuitIR.from_json(
        CircuitIR(logical + 2 + phase_work, native).to_json()
    )
    native_inverse = cost_function_phasing_ir(
        {"data": data},
        {"cost": cost},
        {"phase_grad": grad, "phase_work": work} if gradient else {},
        cost_names=("cost",),
        evaluate=evaluate,
        phase=lambda bound: emit_phase(bound, True),
        erase=evaluate,
    )
    captured = import_register_emitter(source)
    emitter = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    bound = BlackBoxPrepare(ports, (), emitter)
    private, slots = emitter.requirements()
    operators = []
    for inverse in (False, True):
        ir = (
            Circuit(logical + private)
            .append(
                bound.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=tuple(range(logical, logical + private)),
                    classical_bits=tuple(range(slots)),
                    adjoint=inverse,
                )
            )
            .to_ir()
        )
        assert isinstance(ir, CircuitIR)
        operators.append(logical_operator(ir, logical))
    native_emitter = CircuitRegisterEmitter(
        ports,
        (
            native_ir,
            CircuitIR.from_json(
                CircuitIR(logical + 2 + phase_work, native_inverse).to_json()
            ),
        ),
    )
    native_operator, native_backward = through_operators(native_emitter)
    torch.testing.assert_close(operators[1], native_backward, atol=1e-11, rtol=0)
    torch.testing.assert_close(operators[0], native_operator, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        operators[1] @ native_operator,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    assert Counter(g.name for g in captured.snapshots[0].instructions) == Counter(
        g.name for g in native_ir.instructions
    )
    evaluation = ComponentCost("CopyCost", 4, GateCost(clifford=2))
    phase_model = (
        source_qvr_phase_gradient_cost if gradient else source_qvr_zpow_cost
    )(2, 2, coefficient, 0.2)
    model = ComponentCost(
        "PhasingViaCostFunction",
        logical,
        calls=(
            CostCall(evaluation, bystander_qubits=b),
            CostCall(phase_model, bystander_qubits=2),
            CostCall(evaluation, bystander_qubits=b),
        ),
    )
    assert_source_model(model, source)
    if not gradient:
        phases = torch.exp(
            2j * pi * coefficient * torch.arange(4, dtype=torch.float64) / 4
        )
        torch.testing.assert_close(
            native_operator, torch.diag(phases), atol=1e-11, rtol=0
        )
    else:
        assert fmt is not None
        assert b == fmt.gradient_width
        fourier = torch.exp(
            -2j * pi * torch.arange(1 << b, dtype=torch.float64) / (1 << b)
        ) / sqrt(1 << b)
        for x in range(4):
            initial = torch.zeros(1 << logical, dtype=torch.complex128)
            initial[x << b : (x + 1) << b] = fourier
            result = (native_operator @ initial).reshape(4, 1 << b)
            amplitudes = result @ fourier.conj()
            torch.testing.assert_close(
                result, amplitudes[:, None] * fourier[None, :], atol=1e-11, rtol=0
            )
            ideal = torch.exp(
                torch.tensor(2j * pi * coefficient * x / 4, dtype=torch.complex128)
            )
            assert abs(amplitudes[x] - ideal) < 0.2


@pytest.mark.parametrize("width,gradient", [(1, False), (2, False), (1, True)])
def test_original_square_cost_factories_have_explicit_native_value_lowering(
    source_module: Any, width: int, gradient: bool
) -> None:
    import attrs
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic import Square

    from flagquantum.compiler.registers.arithmetic.integer_square import (
        integer_square_ir,
        integer_square_workspace_size,
    )
    from tests.unit.test_phase_gradient_source_binding import through_operators

    root = (
        source_module._square_via_phase_gradient
        if gradient
        else source_module._square_via_zpow_phasing
    ).make()
    phase = attrs.evolve(
        root.phase_oracle,
        cost_reg=Register("result", QFxp(2 * width, 2 * width)),
        gamma=0.375,
        eps=0.2,
    )
    source = attrs.evolve(root, cost_eval_oracle=Square(width), phase_oracle=phase)
    b = int(phase.b_grad) if gradient else 0
    logical = width + b
    fmt = qvr_phase_format(2 * width, 2 * width, 0.375, 0.2) if gradient else None
    terms = (
        scaled_phase_add_terms(
            2 * width,
            2 * width,
            b,
            0.375,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
        if fmt
        else ()
    )
    pw = max(
        (
            phase_add_workspace_size(
                len(tuple(range(2 * width))[part]), b, right_shift=shift
            )
            for part, shift in terms
        ),
        default=0,
    )
    sw = integer_square_workspace_size(width)
    result = tuple(range(logical, logical + 2 * width))
    square_work = tuple(range(logical + 2 * width, logical + 2 * width + sw))
    phase_work = tuple(range(logical + 2 * width + sw, logical + 2 * width + sw + pw))

    def evaluate(ports: RegisterPorts) -> tuple[Instruction, ...]:
        return integer_square_ir(ports["a"], ports["result"], ports["square_work"])

    def build(inverse: bool) -> CircuitIR:
        def phasing(ports: RegisterPorts) -> tuple[Instruction, ...]:
            if gradient:
                return qvr_phase_gradient_ir(
                    ports["result"],
                    ports["phase_grad"],
                    ports["phase_work"],
                    tuple(range(pw)),
                    cost_fraction_bits=2 * width,
                    coefficient=0.375,
                    epsilon=0.2,
                    adjoint=inverse,
                )
            return qvr_zpow_ir(
                ports["result"],
                cost_fraction_bits=2 * width,
                coefficient=0.375,
                epsilon=0.2,
                adjoint=inverse,
            )

        gates = cost_function_phasing_ir(
            {"a": tuple(range(width))},
            {"result": result, "square_work": square_work},
            (
                {"phase_grad": tuple(range(width, logical)), "phase_work": phase_work}
                if gradient
                else {}
            ),
            cost_names=("result",),
            evaluate=evaluate,
            phase=phasing,
            erase=evaluate,
        )
        return CircuitIR.from_json(
            CircuitIR(logical + 2 * width + sw + pw, gates).to_json()
        )

    native = (build(False), build(True))
    captured = import_register_emitter(source)
    owned = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    expected = through_operators(owned)
    actual = through_operators(CircuitRegisterEmitter(owned.layout, native))
    for source_op, native_op, source_ir, native_ir in zip(
        expected, actual, owned.snapshots, native, strict=True
    ):
        torch.testing.assert_close(source_op, native_op, atol=1e-11, rtol=0)
        assert Counter(g.name for g in source_ir.instructions) == Counter(
            g.name for g in native_ir.instructions
        )


def test_unmodified_original_five_bit_direct_factory_complete_complex_state(
    source_module: Any,
) -> None:
    from cmath import exp

    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.sparse_register_execution import execute_sparse_register

    source = source_module._square_via_zpow_phasing.make()
    emitter = import_register_emitter(source)
    ir = CircuitIR.from_json(emitter.snapshots[0].to_json())
    counts = Counter(g.name for g in ir.instructions)
    assert counts["t"] + counts["tdg"] == 160
    assert counts["cx"] == 394 and counts["measure"] == 40 and counts["phase"] == 10
    assert (
        certify_erasure_histories(
            ir.instructions, n_wires=ir.n_wires, zero_wires=tuple(range(5, ir.n_wires))
        )
        == 40
    )
    norm = sqrt(sum(abs(1 + 0.07j * a) ** 2 for a in range(32)))
    initial = {a << (ir.n_wires - 5): (1 + 0.07j * a) / norm for a in range(32)}
    expected = {
        a
        << (ir.n_wires - 5): (1 + 0.07j * a)
        / norm
        * exp(2j * pi * 0.1234 * a * a / 1024)
        for a in range(32)
    }
    for seed in (0, 7):
        actual = execute_sparse_register(ir, dict(initial), seed=seed)
        assert (
            max(
                abs(actual.get(i, 0) - expected.get(i, 0))
                for i in actual.keys() | expected.keys()
            )
            < 1e-11
        )


def test_original_gradient_factory_captures_without_expanding_statevector(
    source_module: Any,
) -> None:
    source = source_module._square_via_phase_gradient.make()
    emitter = import_register_emitter(source)
    assert emitter.layout == (("a", 5), ("phase_grad", 32))
    for ir in emitter.snapshots:
        counts = Counter(g.name for g in ir.instructions)
        assert counts["t"] + counts["tdg"] == 1636
        assert counts["cx"] == 4062 and counts["measure"] == 409
        assert counts["phase"] == 0 and counts["any"] == 0
        assert ir.n_wires < 98
