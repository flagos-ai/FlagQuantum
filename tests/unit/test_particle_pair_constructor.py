"""The standalone particle-pair source constructor retains its model-only body."""

import sys
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_ij_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._first_quantization_component_library import (
    first_quantization_component_factories,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def original_pair_files() -> None:
    module = import_module("qualtran.bloqs.chemistry.pbc.first_quantization.prepare")
    assert module.__file__ is not None
    for name, expected in (
        ("prepare.py", "aba6fd476007748acf99947caba2dcc21e7843be"),
        ("prepare_test.py", "f07ea4087daeb8dae351ed6e8061345b549d0ee6"),
    ):
        raw = (Path(module.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.mark.parametrize("eta,precision", ((1, 8), (2, 1), (10, 8)))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_complete_original_pair_constructor_wire_model_and_physical_refusal(
    eta: int, precision: int, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.pbc.first_quantization import (
        prepare,
        select_and_prepare,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = prepare.UniformSuperpostionIJFirstQuantization(eta, precision)
    duplicate = select_and_prepare.UniformSuperpostionIJFirstQuantization(
        eta, precision
    )
    library = _library(source, depth)
    assert library.entries[0].declaration.name == (
        "qualtran.bloqs.chemistry.pbc.first_quantization.prepare.UniformSuperpostionIJFirstQuantization"
    )
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        graph = resolve_component_program(
            library, 0, first_quantization_component_factories()
        )
    assert not verify_program_record(graph)
    assert not graph.functions[0].blocks
    expected = resolve_component_program(
        _library(duplicate, depth), 0, first_quantization_component_factories()
    )
    assert graph == expected
    assert bool(graph.functions[0].registers) == (eta != 1)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    inputs = tuple(
        tuple(range(4 * i, 4 * i + int(r.bitsize)))
        for i, r in enumerate(source.signature)
    )
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, inputs)
    model = estimate_component_resources(source_prepare_ij_cost(eta, precision))
    original = get_cost_value(source, QECGatesCost())
    assert (
        model.gates.toffoli
        == original.toffoli
        == 7 * (eta - 1).bit_length() + 4 * precision - 18
    )
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert original == get_cost_value(duplicate, QECGatesCost())
