"""Window transformation needs neither Algorithms nor numerical/SDK execution."""

import builtins
from collections.abc import Mapping, Sequence
from types import ModuleType

import pytest

from flagquantum.compiler.formats.legacy_phase_window_factories import (
    phase_window_library_factories,
)
from flagquantum.compiler.registers.phase.window_component import (
    rectangular_window_program,
)
from flagquantum.core.ir.program import ProgramRecord

pytestmark = pytest.mark.unit


def test_constructor_resolution_without_algorithm_numeric_or_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.startswith(
            ("qualtran", "cirq", "flagquantum.algorithms", "flagquantum.simulation")
        ):
            raise AssertionError("Window Compiler imported a forbidden owner: " + name)
        return original(name, globals, locals, fromlist, level)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded)
        factories = phase_window_library_factories()
        rectangular = factories[
            "qualtran.bloqs.phase_estimation.qpe_window_state.RectangularWindowState"
        ]("window", None, {"bitsize": 3})
        kaiser = factories[
            "qualtran.bloqs.phase_estimation.kaiser_window_state.KaiserWindowState"
        ]("kaiser", None, {"bitsize": 3, "alpha": 2})
    assert isinstance(rectangular, ProgramRecord)
    assert rectangular.functions[0].blocks
    assert not isinstance(kaiser, ProgramRecord)
    assert not kaiser.blocks


def test_algorithm_consumer_accepts_replaced_shared_program_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.algorithms.primitives import phase_estimation_window as consumer

    expected = rectangular_window_program(3)
    seen: list[object] = []

    def replacement(width: object) -> ProgramRecord:
        seen.append(width)
        return expected

    monkeypatch.setattr(consumer, "_rectangular_window_program", replacement)
    assert consumer.rectangular_window_program(3) is expected
    assert seen == [3]
