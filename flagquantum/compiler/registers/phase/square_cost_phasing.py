"""Phase an input using its unsigned integer square as the QVR cost."""

from ....core.ir import Instruction
from ..arithmetic.integer_square import integer_square_ir
from .cost_function_phasing import BoundStage, RegisterPorts, cost_function_phasing_ir
from .qvr_phase_gradient import qvr_phase_gradient_ir
from .qvr_zpow import qvr_zpow_ir


def _phase_square(
    source: tuple[int, ...],
    result: tuple[int, ...],
    square_work: tuple[int, ...],
    square_bit: int,
    phase: BoundStage,
    extras: RegisterPorts,
) -> tuple[Instruction, ...]:
    def square(ports: RegisterPorts) -> tuple[Instruction, ...]:
        return integer_square_ir(
            ports["a"],
            ports["result"],
            ports["square_work"],
            measurement_bit=square_bit,
        )

    return cost_function_phasing_ir(
        {"a": source},
        {"result": result, "square_work": square_work},
        extras,
        cost_names=("result",),
        evaluate=square,
        phase=phase,
        erase=square,
    )


def square_cost_zpow_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    square_work: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    square_bit: int = 0,
) -> tuple[Instruction, ...]:
    """Compute a square, phase its unsigned fixed-point interpretation and erase."""

    def phase(ports: RegisterPorts) -> tuple[Instruction, ...]:
        return qvr_zpow_ir(
            ports["result"],
            cost_fraction_bits=cost_fraction_bits,
            coefficient=coefficient,
            epsilon=epsilon,
        )

    return _phase_square(source, result, square_work, square_bit, phase, {})


def square_cost_gradient_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    square_work: tuple[int, ...],
    gradient: tuple[int, ...],
    addition_work: tuple[int, ...],
    addition_bits: tuple[int, ...],
    *,
    cost_fraction_bits: int,
    coefficient: float,
    epsilon: float,
    square_bit: int = 0,
) -> tuple[Instruction, ...]:
    """Kick back the square cost through a caller-owned negative gradient state.

    The result and both workspace registers return to zero. The gradient
    register remains available for reuse. Classical slots can be reused
    across the sequential square and scaled-add stages.
    """

    def phase(ports: RegisterPorts) -> tuple[Instruction, ...]:
        return qvr_phase_gradient_ir(
            ports["result"],
            ports["phase_grad"],
            ports["addition_work"],
            addition_bits,
            cost_fraction_bits=cost_fraction_bits,
            coefficient=coefficient,
            epsilon=epsilon,
        )

    return _phase_square(
        source,
        result,
        square_work,
        square_bit,
        phase,
        {"phase_grad": gradient, "addition_work": addition_work},
    )
