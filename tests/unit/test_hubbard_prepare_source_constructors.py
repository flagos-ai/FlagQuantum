"""Verify complete Hubbard PREPARE source construction and active packed paths."""

import sys
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
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
from flagquantum.compiler.formats.legacy_hubbard_prepare_factories import (
    hubbard_prepare_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._packed_oracle_component_library import (
    packed_oracle_library_factories,
)
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_stored_graph import (
    original_hubbard_state,
    source_terminals,
)
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import prepare_hubbard

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )
    for filename, digest in (
        ("prepare_hubbard.py", "cfdbf20b0baf5ef1f5fa5a7e21826d7f51f5aec5"),
        ("prepare_hubbard_test.py", "ef956c7d0b68de1a5805d76f2c6dd4144adc9485"),
        ("hubbard_model.ipynb", "11862cbbb6894da65d4636f06b9e8c654a1d02cc"),
    ):
        raw = (Path(prepare_hubbard.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return prepare_hubbard


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize(
    "t,u", [(1.0, 4.0), (0.37, 0.27), (0.0, 4.0), (1.0, 0.0), (-1.0, -4.0)]
)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_complete_constructor_metadata_calls_angles_and_exact_inverse(
    source_module: Any, dimension: int, t: float, u: float, inverse: bool, depth: int
) -> None:
    source = source_module.PrepareHubbard(dimension, dimension, t, u)
    if inverse:
        source = source.adjoint()
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    record = resolve_component_program(library, 0, packed_oracle_library_factories())
    assert verify_program_record(record) == ()
    pairs = compare_layers(record, source, source_terminals(record))
    for name, child in pairs.items():
        rotation = child.subbloq if type(child).__name__ == "Controlled" else child
        if type(rotation).__name__ != "Ry":
            continue
        count = len(next(f for f in record.functions if f.name == name).registers or ())
        gates = bind_register_program(
            record, tuple((i,) for i in range(count)), entry=name
        ).circuit.instructions
        assert len(gates) == 1 and gates[0].params["theta"] == float(rotation.angle)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("t,u", [(1.0, 4.0), (0.37, 0.27), (0.0, 4.0), (1.0, 0.0)])
def test_floor_normalization_and_endpoint_resource_categories(
    source_module: Any, dimension: int, t: float, u: float
) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = source_module.PrepareHubbard(dimension, dimension, t, u)
    operation = PrepareHubbard(dimension, t, u)
    assert operation.alpha == source.l1_norm_of_coeffs
    model = estimate_component_resources(operation.source_resource_model())
    expected = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model.gates, category) == getattr(expected, category)
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("t,u", [(0.37, 0.27), (0.0, 4.0), (-1.0, -4.0)])
def test_all_small_raw_histories_absolute_complex_amplitudes_and_logical_junk(
    source_module: Any, t: float, u: float
) -> None:
    source = source_module.PrepareHubbard(2, 2, t, u)
    logical, work = 10, 2
    rng = np.random.default_rng(2084)
    initial = rng.normal(size=1 << logical) + 1j * rng.normal(size=1 << logical)
    initial /= np.linalg.norm(initial)
    expected = original_hubbard_state(2, initial, t=t, u=u)
    inputs = ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9))
    clean = (10, 11)
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        record = resolve_component_program(
            _library(original, 0), 0, packed_oracle_library_factories()
        )
        bound = bind_register_program(record, inputs, clean=clean)
        assert bound.outputs == inputs
        events = certify_erasure_histories(
            bound.circuit.instructions,
            n_wires=12,
            zero_wires=clean,
            check_clean_output=False,
        )
        assert (
            events
            == Counter(g.name for g in bound.circuit.instructions)["measure"]
            == 3
        )
        state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
        desired = torch.zeros_like(state)
        state[::4] = torch.as_tensor(expected if inverse else initial)
        desired[::4] = torch.as_tensor(initial if inverse else expected)
        for history in product((0, 1), repeat=events):
            actual = forced_branch(state, bound.circuit.instructions, history)
            torch.testing.assert_close(
                actual, desired / 2 ** (events / 2), atol=3e-11, rtol=0
            )
            assert actual.reshape(-1, 4)[:, 1:].abs().max() < 3e-11


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_source_free_constructor_and_normal_append(
    source_module: Any, strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_module.PrepareHubbard(2, 2, 0.37, 0.27)
    library = _library(source, 0)
    initial = np.zeros(1024, dtype=np.complex128)
    initial[0], initial[3] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    expected = original_hubbard_state(2, initial, t=0.37, u=0.27)
    for module in ("qualtran", "cirq", "kickmix", "quimb"):
        monkeypatch.setitem(sys.modules, module, None)
    record = resolve_component_program(library, 0, packed_oracle_library_factories())
    bound = bind_register_program(
        record, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9)), clean=(10, 11)
    )
    state = torch.zeros(4096, dtype=torch.complex128)
    state[::4] = torch.as_tensor(initial)
    desired = torch.zeros_like(state)
    desired[::4] = torch.as_tensor(expected)
    circuit = DynamicCircuit(12, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(12, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=11, strategy=strategy)
    assert result.statistics["measurement_count"] == 24
    for result_state in result.final_states:
        torch.testing.assert_close(result_state.flatten(), desired, atol=3e-11, rtol=0)


def test_original_factory_singleton_and_approved_symbolic_body(
    source_module: Any,
) -> None:
    source = source_module._prep_hubb.make()
    assert (source.x_dim, source.y_dim, source.t, source.u) == (4, 4, 1.0, 4.0)
    assert source_module._PREPARE_HUBBARD.examples == (source_module._prep_hubb,)
    record = resolve_component_program(
        _library(source, 0), 0, packed_oracle_library_factories()
    )
    compare_layers(record, source, source_terminals(record))
    symbol = sympy.Symbol("t")
    singleton = source_module.PrepareHubbard(1, 1, 1.0, 4.0)
    record = resolve_component_program(
        _library(singleton, 0), 0, packed_oracle_library_factories()
    )
    assert not record.functions[0].blocks
    symbolic = source_module.PrepareHubbard(2, 2, symbol, 4.0)
    assert (
        resolve_component_program(
            _library(symbolic, 0), 0, packed_oracle_library_factories()
        )
        .functions[0]
        .blocks
    )
    with pytest.raises(ValueError, match="nonnegative"):
        PrepareHubbard(2, -1.0, -4.0)
    invalid = source_module.PrepareHubbard(2, 2, 0.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        invalid.decompose_bloq()


def test_integer_coefficients_are_not_rounded_before_normalization() -> None:
    seen: list[tuple[int, object, object]] = []

    def capture(dimension: int, t: Any, u: Any) -> None:
        seen.append((dimension, t, u))

    t, u = 2**60 + 1, 2**59 + 3
    factory = hubbard_prepare_factories(angles=capture)[
        "qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard"
    ]
    factory("main", None, {"x_dim": 2, "y_dim": 2, "t": t, "u": u})
    assert seen == [(2, t, u)] and type(seen[0][1]) is int
