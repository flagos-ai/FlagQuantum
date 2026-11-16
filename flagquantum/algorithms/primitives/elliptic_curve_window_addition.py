# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECWindowAddR.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add_r.py

"""Load point multiples and tangents, add coherently and erase the lookup banks."""

from dataclasses import dataclass
from functools import cached_property

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler.registers.lookup.qroam_clean_erasure import (
    qroam_clean_measurement_erase_ir,
)
from ...compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    optimal_qroam_clean_log_block_size,
    source_qroam_clean_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ...dynamic import DynamicCircuit
from ...simulation.montgomery_values import (
    integer_to_montgomery_value,
    montgomery_to_integer_value,
)
from .elliptic_curve_addition import EllipticCurveAddition
from .elliptic_curve_fixed_point import fixed_point_addition_signature
from .elliptic_curve_point import ECPoint


def windowed_point_addition_source_graph() -> SymbolicCallGraph[str]:
    """Declare the source lookup, curve addition and lookup erase calls."""
    root = "ECWindowAddR"
    return symbolic_call_graph(
        root, {root: {"QROAMClean": 1, "ECAdd": 1, "QROAMCleanAdjointWrapper": 1}}
    )


def windowed_point_table(width: int, window: int, point: ECPoint) -> QroamCleanTable:
    """Construct the locked three datasets in increasing big-endian address order."""
    if type(width) is not int or width < 1 or type(window) is not int or window < 1:
        raise ValueError("Windowed point tables require positive widths")
    modulus = int(point.mod)
    rows = 1 << window
    data_x, data_y, data_tangent = [0], [0], [0]
    multiple = point
    for _ in range(1, rows):
        x, y = int(multiple.x), int(multiple.y)
        denominator = (2 * y) % modulus
        tangent = (
            ((3 * x * x + int(multiple.curve_a)) * pow(denominator, -1, modulus))
            % modulus
            if denominator
            else 0
        )
        data_x.append(int(integer_to_montgomery_value(x, width, modulus)))
        data_y.append(int(integer_to_montgomery_value(y, width, modulus)))
        data_tangent.append(int(integer_to_montgomery_value(tangent, width, modulus)))
        multiple = multiple + point
    block = optimal_qroam_clean_log_block_size(rows, 3 * width)
    return QroamCleanTable(
        (rows,), (block,), (tuple(data_x), tuple(data_y), tuple(data_tangent))
    )


def windowed_point_classical_values(
    width: int, point: ECPoint, address: int, x: int, y: int
) -> tuple[int, int, int]:
    """Evaluate the numeric source map independently of gate-domain admission."""
    modulus = int(point.mod)
    current = ECPoint(
        int(montgomery_to_integer_value(x, width, modulus)),
        int(montgomery_to_integer_value(y, width, modulus)),
        modulus,
        point.curve_a,
    )
    result = current + address * point
    return (
        address,
        int(integer_to_montgomery_value(int(result.x), width, modulus)),
        int(integer_to_montgomery_value(int(result.y), width, modulus)),
    )


@dataclass(frozen=True)
class _BoundEllipticCurveWindowAddition:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError(
                "Windowed elliptic-curve addition requires a DynamicCircuit"
            )
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurveWindowAddition:
    """Bind an address, a Montgomery point, and clear the matching table banks.

    The address is big-endian and unchanged. The integer point defining the
    table is supplied in raw affine coordinates. The six-stage child inherits
    the locked arithmetic and flag-release promises; its residual flags on
    out-of-promise inputs remain observable in caller-owned workspace.
    """

    width: int
    point: ECPoint
    window: int
    multiplication_window: int = 1

    def __post_init__(self) -> None:
        if type(self.window) is not int or self.window < 1:
            raise ValueError(
                "Windowed curve addition requires a positive address width"
            )
        EllipticCurveAddition(
            self.width, int(self.point.mod), self.multiplication_window
        )

    @property
    def source_signature(self) -> Signature:
        return fixed_point_addition_signature(self.width, self.window)

    @cached_property
    def table(self) -> QroamCleanTable:
        return windowed_point_table(self.width, self.window, self.point)

    def classical_values(self, address: int, x: int, y: int) -> tuple[int, int, int]:
        """Evaluate the source affine callback on the stored coordinate words."""
        if type(address) is not int or not 0 <= address < 1 << self.window:
            raise ValueError("Address must fit the declared window")
        return windowed_point_classical_values(self.width, self.point, address, x, y)

    def workspace_requirements(self) -> tuple[int, int]:
        """Count retained banks, shared scratch and distinct bank readout slots."""
        n = self.width
        count = self.table.block_shape[0] * 3 * n
        traversal = self.table.workspace_size(0, (self.window,))
        add_q, add_c = EllipticCurveAddition(
            n, int(self.point.mod), self.multiplication_window
        ).workspace_requirements()
        return count + max(add_q, traversal), count + max(add_c, traversal)

    def make_on(
        self,
        *,
        ctrl: tuple[int, ...],
        x: tuple[int, ...],
        y: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundEllipticCurveWindowAddition:
        n, k = self.width, self.window
        q, c = self.workspace_requirements()
        if len(ctrl) != k or len(x) != n or len(y) != n:
            raise ValueError(
                "Windowed curve addition requires the declared address and coordinate widths"
            )
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Windowed curve addition requires exact workspace sizes")
        wires = (*ctrl, *x, *y, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Windowed curve addition requires distinct nonnegative wires and slots"
                )
        ctrl = tuple(range(k))
        x = tuple(range(k, k + n))
        y = tuple(range(k + n, k + 2 * n))
        work = tuple(range(k + 2 * n, len(wires)))
        cells = self.table.block_shape[0]
        count = 3 * n * cells
        a, b, tangent = (
            tuple(
                work[(dataset * cells + i) * n : (dataset * cells + i + 1) * n]
                for i in range(cells)
            )
            for dataset in range(3)
        )
        scratch = work[count:]
        bank_slots = classical_bits[-count:]
        traversal = self.table.workspace_size(0, (k,))
        lookup_args = (
            self.table,
            (),
            (ctrl,),
            (a, b, tangent),
            scratch[:traversal],
            classical_bits[:traversal],
        )
        add = EllipticCurveAddition(n, int(self.point.mod), self.multiplication_window)
        add_q, add_c = add.workspace_requirements()
        circuit = DynamicCircuit(len(wires))
        load = CircuitIR(len(wires), source_qroam_clean_ir(*lookup_args)).validate()
        circuit.compose(load)
        circuit.append(
            add.make_on(
                a=a[0],
                b=b[0],
                x=x,
                y=y,
                lam_r=tangent[0],
                workspace=scratch[:add_q],
                classical_bits=classical_bits[:add_c],
            )
        )
        erase = CircuitIR(
            len(wires), qroam_clean_measurement_erase_ir(*lookup_args, bank_slots)
        ).validate()
        circuit.compose(erase)
        return _BoundEllipticCurveWindowAddition(wires, circuit.to_ir().instructions)
