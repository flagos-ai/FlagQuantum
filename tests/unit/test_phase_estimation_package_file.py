"""Map every frozen phase-estimation package entry to its domain-owned module."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

_OWNERS = {
    "KaiserWindowState": ("kaiser_window_state", "kaiser_window", "KaiserWindowState"),
    "LPResourceState": ("lp_resource_state", "luis_perina_state", "LuisPerinaState"),
    "RectangularWindowState": (
        "qpe_window_state",
        "phase_estimation_window",
        "RectangularWindowState",
    ),
    "QubitizationQPE": (
        "qubitization_qpe",
        "qubitization_phase_estimation",
        "QubitizationPhaseEstimation",
    ),
    "TextbookQPE": (
        "text_book_qpe",
        "textbook_phase_estimation",
        "TextbookPhaseEstimation",
    ),
}


def test_complete_locked_package_has_exactly_five_owned_entries() -> None:
    original = import_module("qualtran.bloqs.phase_estimation")
    assert original.__file__ is not None
    raw = Path(original.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "3f3e49797e3358ec6c581fac78b3a5bc75c9cd97"
    )
    statements = ast.parse(raw).body
    assert len(statements) == len(_OWNERS) == 5
    seen = set()
    for statement in statements:
        assert isinstance(statement, ast.ImportFrom)
        assert statement.level == 0
        (identifier,) = statement.names
        assert identifier.asname is None and identifier.name not in seen
        seen.add(identifier.name)
        source_owner, native_owner, native_name = _OWNERS[identifier.name]
        assert statement.module == f"qualtran.bloqs.phase_estimation.{source_owner}"
        source = import_module(statement.module)
        assert getattr(original, identifier.name) is getattr(source, identifier.name)
        native = import_module(f"flagquantum.algorithms.primitives.{native_owner}")
        entry = getattr(native, native_name)
        assert entry.__module__ == native.__name__
        assert callable(entry)
    assert seen == set(_OWNERS)


@pytest.mark.parametrize("source_name", tuple(_OWNERS)[:3])
def test_imported_window_factories_preserve_complete_registers_and_coefficients(
    source_name: str,
) -> None:
    source_owner, native_owner, native_name = _OWNERS[source_name]
    source_type = getattr(
        import_module(f"qualtran.bloqs.phase_estimation.{source_owner}"), source_name
    )
    native_type = getattr(
        import_module(f"flagquantum.algorithms.primitives.{native_owner}"), native_name
    )
    args = (3, 1.0) if source_name == "KaiserWindowState" else (3,)
    source, native = source_type(*args), native_type(*args)
    assert native.m_bits == source.m_bits == 3
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.signature
    ]
    if source_name == "KaiserWindowState":
        np.testing.assert_allclose(
            native.kaiser_state_coeff, source.kaiser_state_coeff, atol=1e-15, rtol=0
        )
