"""Shared fixtures for integration tests."""

import pytest

from flagquantum.synth import _program as _synthesis_registers


@pytest.fixture
def legacy_register_selection(monkeypatch):
    """Exercise historical register scheduling invariants explicitly."""
    selector = _synthesis_registers.select_register_program

    def select(*args, **kwargs):
        kwargs["include_boolean"] = False
        return selector(*args, **kwargs)

    monkeypatch.setattr(_synthesis_registers, "select_register_program", select)
