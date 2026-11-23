# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/circuit/subs/unary_iteration.pybind.cc/.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Own a reusable compile-time unary traversal pass over shared clean workspace."""

from operator import index as integer_index
from types import TracebackType
from typing import Literal, SupportsIndex

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseOperation
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..layout.workspace import RegisterWorkspace, WorkspaceLease
from ..primitives._mixed_gates import _XBasisTarget
from ..primitives.operand_binding import bind_operand
from .recursive_unary_iteration import UnaryAddressCursor


class UnaryIterationPass:
    """Emit source cursor transitions with explicit temporary allocation lifetime."""

    def __init__(
        self,
        operations: list[ReversiblePhaseOperation],
        workspace: RegisterWorkspace,
        address: tuple[int, ...],
        address_values: object = None,
        *,
        control: object = True,
        measurement_bit: int = 0,
    ) -> None:
        operand = bind_operand(control)
        if isinstance(operand, _XBasisTarget):
            raise ValueError(
                "Unary control must be a Boolean, quantum wire or classical slot"
            )
        self.control = operand
        quantum_control = (self.control,) if type(self.control) is int else ()
        _validate_registers(address, quantum_control)
        if len(address) > 63:
            raise ValueError("len(address) > 63")
        if address_values is not None and not isinstance(address_values, range):
            raise ValueError("address_values must be None or a range")
        if address_values is not None and address_values.step != 1:
            raise ValueError(
                "address_values must have step 1; visit other patterns with move_to"
            )
        if type(measurement_bit) is not int or measurement_bit < 0:
            raise ValueError("Unary erasure requires a nonnegative result slot")
        if (
            isinstance(self.control, ClassicalFieldCoefficient)
            and self.control.slot == measurement_bit
        ):
            raise ValueError(
                "Unary guard and erasure result require distinct classical slots"
            )
        self.address = address
        self.address_values = (
            address_values if address_values is not None else range(1 << len(address))
        )
        self.operations = operations
        self.calls: list[ComponentCall] = []
        self.workspace = workspace
        self.measurement_bit = measurement_bit
        self._lease: WorkspaceLease | None = None
        self._cursor: UnaryAddressCursor | None = None
        self._position = 1 << len(address)
        self._next = self.address_values.start
        self._iterating = False

    def _check_open(self) -> WorkspaceLease:
        if self._lease is None:
            raise ValueError(
                "The unary iteration cursor is not open. Use a with block to open a pass"
            )
        return self._lease

    def __enter__(self) -> "UnaryIterationPass":
        if self._lease is not None:
            raise ValueError("The unary iteration cursor is already open")
        count = 1 if self.control is False else max(1, len(self.address))
        lease = self.workspace.allocate(count)
        try:
            cursor = None
            if self.control is not False:
                cursor = UnaryAddressCursor(
                    self.address,
                    lease.wires,
                    control=self.control if type(self.control) is int else None,
                    measurement_bit=self.measurement_bit,
                )
        except ValueError:
            self.workspace.free(lease)
            raise
        self._lease = lease
        self._cursor = cursor
        self._position = 1 << len(self.address)
        self._next = self.address_values.start
        if isinstance(self.control, ClassicalFieldCoefficient):
            self.operations.append(
                ReversiblePhaseOperation(
                    "push_condition", (), condition_bit=self.control.slot
                )
            )
        return self

    def _append_move(self, operations: tuple[ReversiblePhaseOperation, ...]) -> None:
        if operations:
            start = len(self.operations)
            self.operations.extend(operations)
            self.calls.append(
                ComponentCall("unary_iteration", start, len(self.operations))
            )

    def close(self) -> None:
        self._iterating = False
        if self._lease is None:
            return
        if self._cursor is not None:
            self._append_move(self._cursor.close())
            self._cursor = None
        if isinstance(self.control, ClassicalFieldCoefficient):
            self.operations.append(ReversiblePhaseOperation("pop_condition", ()))
        self.workspace.free(self._lease)
        self._lease = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    @property
    def match_qubit(self) -> int:
        return self._check_open().wires[0]

    @property
    def cur_address_value(self) -> int:
        self._check_open()
        return self._position

    def move_to(self, value: SupportsIndex) -> None:
        position = integer_index(value)
        self._check_open()
        self._position = position
        if self._cursor is not None:
            limit = 1 << len(self.address)
            self._append_move(
                self._cursor.move_to(position if 0 <= position < limit else limit)
            )

    def __iter__(self) -> "UnaryIterationPass":
        self._check_open()
        if self._iterating:
            raise ValueError("The unary iteration cursor is already being iterated")
        self._next = self.address_values.start
        return self

    def __next__(self) -> tuple[int, int]:
        self._check_open()
        if self._cursor is None or self._next >= self.address_values.stop:
            self._iterating = False
            raise StopIteration
        self._iterating = True
        value = self._next
        self._next += 1
        self.move_to(value)
        return value, self.match_qubit

    def __repr__(self) -> str:
        return f"UnaryIterationPass(address={self.address!r}, address_values={self.address_values!r}, control={self.control!r})"

    def __copy__(self) -> "UnaryIterationPass":
        raise TypeError("Unary iteration passes own a unique workspace lifetime")

    def __deepcopy__(self, memo: dict[int, object]) -> "UnaryIterationPass":
        raise TypeError("Unary iteration passes own a unique workspace lifetime")
