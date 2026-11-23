"""Route unchanged locked source tests to native numerical session execution.

This opt-in pytest plugin converts source-owned external objects immediately.
The original test file and every assertion remain unchanged. Production modules
never import the reference extension.
"""

from typing import Any

import kickmix as km
import pytest

from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession


def _translate(value: Any) -> object:
    for kind, external in (
        (OperandKind.QUANTUM, km.q),
        (OperandKind.CLASSICAL, km.b),
        (OperandKind.REGISTER, km.r),
    ):
        if isinstance(value, external):
            return OperandIdentity(kind, value.id)
    if isinstance(value, (list, tuple, km.array)):
        return [_translate(item) for item in value]
    return value


class Replacement:
    """Keep original external values at the test adapter boundary."""

    def __init__(self, *args: Any, debug_print: Any = True, **kwargs: Any) -> None:
        self.native = PhysicalBasisSession(*args, debug_print=False, **kwargs)
        self.debug_print = debug_print

    @property
    def batch_size(self) -> int:
        return self.native.batch_size

    @property
    def counting_operations(self) -> bool:
        return self.native.counting_operations

    @property
    def debug_print(self) -> Any:
        return self._debug

    @debug_print.setter
    def debug_print(self, value: Any) -> None:
        if callable(value):

            def callback(target: Any, values: Any, predicate: Any, active: Any) -> None:
                external_target = None
                if target is not None:
                    factory = km.q if target.kind is OperandKind.QUANTUM else km.b
                    external_target = factory(target.value)
                external_predicate = (
                    None if predicate is None else km.b(predicate.value)
                )
                value(external_target, values, external_predicate, active)

            self.native.debug_print = callback
        else:
            self.native.debug_print = value
        self._debug = value

    def do(self, circuit: Any) -> None:
        self.native.do(read_kmx_circuit(str(circuit)))

    def use_same_registers_as(self, circuit: Any) -> None:
        self.native.use_same_registers_as(read_kmx_circuit(str(circuit)))

    def read_across_shots(self, indices: Any, **kwargs: Any) -> object:
        return self.native.read_across_shots(_translate(indices), **kwargs)

    def read_within_shot(self, indices: Any, shot_index: Any, **kwargs: Any) -> object:
        return self.native.read_within_shot(_translate(indices), shot_index, **kwargs)

    def write_across_shots(self, indices: Any, new_value: Any) -> None:
        self.native.write_across_shots(_translate(indices), new_value)

    def write_within_shot(self, indices: Any, shot_index: Any, new_value: Any) -> None:
        self.native.write_within_shot(_translate(indices), shot_index, new_value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if (
            isinstance(item, pytest.Function)
            and item.path.name == "interactive_simulator_test.py"
        ):
            item.module.km.Simulator = Replacement
