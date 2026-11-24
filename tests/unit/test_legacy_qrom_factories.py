"""Resolve complete concrete QROM metadata without losing lookup graph semantics."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
from flagquantum.compiler.registers.lookup.qrom_component import qrom_program
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_qrom_complete_file import native_table
from tests.unit.test_qrom_component import source_table

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def pinned_reference() -> None:
    import kickmix
    from qualtran.bloqs.data_loading import qrom

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(qrom.__file__).parent.parent
    for name, digest in (
        ("data_loading/qrom.py", "79b2f530809bff5d4f8df3793b755b050731079b"),
        ("data_loading/qrom_base.py", "efe1c5e2300642b8a15d282695c1f73402eb084d"),
        ("data_loading/qrom_test.py", "a2dc34d9bd66ab4ab213f18a2655e80f45cadfc9"),
        (
            "multiplexers/unary_iteration_bloq.py",
            "d1f689a99ecccaf9a44983d074875f2dce64ccb6",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize(
    "example", ("_qrom_small", "_qrom_multi_data", "_qrom_multi_dim")
)
def test_original_concrete_examples_retain_complete_metadata(example: str) -> None:
    from qualtran.bloqs.data_loading import qrom

    source = getattr(qrom, example).make()
    record = resolve_component_program(_library(source, 0), 0, qrom_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == tuple(native_table(source).signature)
    assert resolve_component_program(_library(source, 1), 0, qrom_factories()) == record
    compare_layers(record, source, atomic_leaves(record))


@pytest.mark.parametrize(
    "controls,data",
    tuple(product((0, 1, 2, 3), ((0, 1, 3, 2), (2, 2, 3, 3), (1, 1, 1, 1)))),
)
def test_entry_replacement_exact_calls_feedback_dirty_outputs_and_complex_operator(
    controls: int, data: tuple[int, ...]
) -> None:
    table = QromTable((4,), (2,), (2,), ((),), controls, (data,))
    source = source_table(table)
    record = resolve_component_program(_library(source, 0), 0, qrom_factories())
    compare_layers(record, source, atomic_leaves(record))
    logical = controls + 4
    inputs = (
        *((tuple(range(controls)),) if controls else ()),
        tuple(range(controls, controls + 2)),
        tuple(range(controls + 2, logical)),
    )
    clean = tuple(range(logical, logical + table.workspace_size))
    actual = bind_register_program(record, inputs, clean=clean)
    accepted = bind_register_program(qrom_program(table), inputs, clean=clean)
    assert actual == accepted
    if controls <= 1:
        torch.testing.assert_close(
            logical_operator(actual.circuit, logical),
            torch.tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("shape,target_shape", (((2, 2), ()), ((3,), (2,))))
def test_multiaxis_and_target_array_constructors(
    shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    values = (
        np.arange(int(np.prod(shape + target_shape))).reshape(shape + target_shape) % 4
    )
    table = QromTable.from_data(
        values, target_bitsizes=(2,), target_shapes=(target_shape,)
    )
    source = source_table(table)
    record = resolve_component_program(_library(source, 0), 0, qrom_factories())
    assert record.functions[0].registers == tuple(table.signature)
    compare_layers(record, source, atomic_leaves(record))


def test_original_symbolic_serialization_boundary_and_native_shape_owner() -> None:
    from qualtran.bloqs.data_loading.qrom import _qrom_symb

    source = _qrom_symb.make()
    with pytest.raises(ValueError, match="Object arrays"):
        _library(source, 0)
    table = native_table(source)
    record = qrom_program(table)
    assert record.functions[0].registers == tuple(table.signature)
    assert not record.functions[0].blocks


def test_invalid_constructor_metadata_is_rejected() -> None:
    factory = next(iter(qrom_factories().values()))
    with pytest.raises(SerializationError, match="arguments"):
        factory("main", None, {})
    arguments: dict[str, Any] = {
        "data_or_shape": np.array([[0, 1]]),
        "selection_bitsizes": np.array([1]),
        "target_bitsizes": np.array([1]),
    }
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), arguments)
    for value in (np.array([[1]]), np.array([1.5])):
        with pytest.raises(SerializationError, match="(vector|integers)"):
            factory("main", None, arguments | {"target_bitsizes": value})
    with pytest.raises(SerializationError, match="concrete datasets"):
        factory("main", None, arguments | {"data_or_shape": np.array(1)})
    with pytest.raises(SerializationError, match="one target shape"):
        factory("main", None, arguments | {"target_shapes": ()})
