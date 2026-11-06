"""Table-driven inverse distance from Eqs. 36–49 of arXiv:2308.12352."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, localcontext

from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.fixed_point_horner import (
    register_horner_step_ir,
    register_horner_step_requirements,
)
from ...compiler.registers.inverse_sqrt_newton import (
    inverse_sqrt_newton_requirements,
    inverse_sqrt_newton_step_ir,
)
from ...compiler.registers.multi_control import multi_and_ir
from ...compiler.registers.source_qrom import source_qrom_ir, source_qrom_workspace_size
from ...core.ir import Instruction
from .grid_inverse_sqrt import _HIGH_DECIMAL, _LOW_DECIMAL

_DELTA_LOW = Decimal("5.1642030908180720584e-9")
_DELTA_HIGH = Decimal("3.6279794522852781448e-10")


@dataclass(frozen=True)
class _Layout:
    distance: int
    precision: int
    tables: tuple[tuple[int, ...], ...]
    correction: int
    qrom: int
    horner: tuple[int, int]
    newton: tuple[int, int]

    @property
    def storage(self) -> int:
        return (self.distance - 1) + 8 * self.precision + 1 + self.correction

    @property
    def work(self) -> int:
        return self.storage + max(
            self.qrom, self.distance - 1, self.horner[0], self.newton[0]
        )

    @property
    def slots(self) -> int:
        return max(self.qrom, self.distance - 1, self.horner[1], self.newton[1])


def _layout(distance: int, precision: int) -> _Layout:
    if type(distance) is not int or distance < 2:
        raise ValueError(
            "Cubic inverse square root requires at least two distance bits"
        )
    if type(precision) is not int or not 1 <= precision <= 63:
        raise ValueError("Cubic inverse square root requires 1 to 63 fractional bits")
    rows: list[tuple[int, ...]] = [(0, 0, 0, 0, 0)]
    low = tuple(map(Decimal, _LOW_DECIMAL))
    high = tuple(map(Decimal, _HIGH_DECIMAL))
    with localcontext() as context:
        context.prec = 80
        for k in range(distance):
            scale = Decimal(1 << k).sqrt()
            intervals = (
                ((low, _DELTA_LOW), (high, _DELTA_HIGH)) if k else ((low, _DELTA_LOW),)
            )
            for coefficients, delta in intervals:
                row = tuple(
                    int(coefficient * (1 << precision) / scale)
                    for coefficient in coefficients
                )
                correction = int(delta * (1 << (2 * precision)))
                rows.extend([(*row, correction)] * (1 << max(0, k - 1)))
    tables = tuple(zip(*rows, strict=True))
    return _Layout(
        distance,
        precision,
        tables,
        max(1, max(tables[-1]).bit_length()),
        source_qrom_workspace_size(0, (distance,), tables),
        register_horner_step_requirements(distance - 1, precision),
        inverse_sqrt_newton_requirements(
            distance,
            precision,
            source_fraction_bits=0,
            estimate_fraction_bits=precision,
        ),
    )


def grid_cubic_inverse_sqrt_requirements(
    distance_width: int, precision: int
) -> tuple[int, int]:
    """Return clean work and feedback slots for the cubic plus one Newton update."""
    layout = _layout(distance_width, precision)
    return layout.work, layout.slots


def _normalized_argument_ir(
    distance: tuple[int, ...],
    argument: tuple[int, ...],
    scratch: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    gates: list[Instruction] = []
    n = len(distance)
    for k in range(2, n):
        prefix = distance[: n - k]
        values = (0,) * (len(prefix) - 1) + (1,)
        work = scratch[: len(prefix) - 1]
        control = work[-1] if work else prefix[0]
        if work:
            gates.extend(multi_and_ir(prefix, values, work))
        gates.extend(
            Instruction("ccx", (control, source, target))
            for source, target in zip(distance[-(k - 1) :], argument[1:k], strict=True)
        )
        if work:
            gates.extend(
                multi_and_ir(
                    prefix,
                    values,
                    work,
                    classical_bits=bits[: len(work)],
                    uncompute=True,
                )
            )
    return tuple(gates)


def with_grid_cubic_inverse_sqrt_ir(
    distance: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    precision: int,
    consume: Callable[[tuple[int, ...]], tuple[Instruction, ...]],
) -> tuple[Instruction, ...]:
    """Use the refined unsigned fixed-point estimate before erasing all work.

    A zero distance yields zero. The consumer must preserve the estimate.
    """
    layout = _layout(len(distance), precision)
    if len(workspace) != layout.work or len(classical_bits) != layout.slots:
        raise ValueError(
            "Cubic inverse square root requires matching workspace and slots"
        )
    _validate_wires((*distance, *workspace))
    argument = workspace[: len(distance) - 1]
    banks_and_history = workspace[len(argument) : len(argument) + 7 * precision]
    coeff = tuple(
        banks_and_history[i : i + precision] for i in range(0, 4 * precision, precision)
    )
    history = tuple(
        banks_and_history[i : i + precision]
        for i in range(4 * precision, 7 * precision, precision)
    )
    tail = workspace[len(argument) + 7 * precision : layout.storage]
    refined, correction = tail[: precision + 1], tail[precision + 1 :]
    scratch = workspace[layout.storage :]
    load = source_qrom_ir(
        (),
        (distance,),
        (1 << len(distance),),
        tuple((word,) for word in (*coeff, correction)),
        layout.tables,
        scratch[: layout.qrom],
        classical_bits[: layout.qrom],
    )
    normalize = _normalized_argument_ir(distance, argument, scratch, classical_bits)
    steps = tuple(
        zip((coeff[-1], *history[:-1]), reversed(coeff[:-1]), history, strict=True)
    )

    def horner(adjoint: bool) -> tuple[Instruction, ...]:
        return tuple(
            gate
            for previous, coefficient, result in (reversed(steps) if adjoint else steps)
            for gate in register_horner_step_ir(
                argument,
                previous,
                coefficient,
                result,
                scratch[: layout.horner[0]],
                classical_bits[: layout.horner[1]],
                argument_fraction_bits=len(argument),
                adjoint=adjoint,
            )
        )

    refine = inverse_sqrt_newton_step_ir(
        distance,
        history[-1],
        refined,
        scratch[: layout.newton[0]],
        classical_bits[: layout.newton[1]],
        source_fraction_bits=0,
        estimate_fraction_bits=precision,
        correction=correction,
    )
    return (
        *load,
        *normalize,
        *horner(False),
        *refine,
        *consume(refined),
        *refine,
        *horner(True),
        *normalize,
        *load,
    )
