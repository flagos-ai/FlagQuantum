"""Verify that all four locked QFT package entries have native owners."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

from flagquantum.algorithms import primitives
from flagquantum.algorithms.primitives.qft import append_qft, qft

pytestmark = pytest.mark.unit

_SOURCE_BLOB = "cf1da7994d7f70e90ef04b7f1a63f99c1e464902"
# Each source class has a distinct execution and resource contract.
_OWNERS = {
    "ApproximateQFT": ("approximate_qft", "approximate_qft", "approximate_qft_ir"),
    "QFTPhaseGradient": (
        "qft_phase_gradient",
        "qft_phase_gradient",
        "qft_phase_gradient_ir",
    ),
    "QFTTextBook": ("qft_text_book", "qft", "qft"),
    "TwoBitFFFT": ("two_bit_ffft", "fermionic_fourier", "two_bit_fermionic_fourier_ir"),
}


@pytest.mark.ftqc_reference
def test_locked_qft_initializer_covers_exactly_the_native_owners() -> None:
    package = pytest.importorskip("qualtran.bloqs.qft")
    assert package.__file__ is not None
    text = Path(package.__file__).read_bytes()
    assert (
        sha1(b"blob " + str(len(text)).encode() + b"\0" + text).hexdigest()
        == _SOURCE_BLOB
    )
    source = ast.parse(text)
    assert len(source.body) == len(_OWNERS)
    seen = set()
    for statement in source.body:
        assert isinstance(statement, ast.ImportFrom)
        assert statement.level == 1
        (identifier,) = statement.names
        assert identifier.asname is None
        assert identifier.name not in seen
        seen.add(identifier.name)
        source_owner, native_owner, native_name = _OWNERS[identifier.name]
        assert statement.module == source_owner
        source_module = import_module(f"{package.__name__}.{source_owner}")
        assert getattr(package, identifier.name) is getattr(
            source_module, identifier.name
        )
        module = import_module(f"{primitives.__name__}.{native_owner}")
        entry = getattr(module, native_name)
        assert callable(entry)
        assert entry.__module__ == module.__name__
    assert seen == set(_OWNERS)


def test_package_keeps_the_existing_public_fourier_entry_points() -> None:
    assert primitives.qft is qft
    assert primitives.append_qft is append_qft
    assert {"qft", "append_qft"} <= set(primitives.__all__)
    for source_name, (_, _, native_name) in _OWNERS.items():
        assert source_name not in primitives.__all__
        if native_name != "qft":
            assert native_name not in primitives.__all__
