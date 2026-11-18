"""Complete source conversion declaration and its real resource consumer."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.algorithms import first_quantization_resource_model as resources
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.bits.signed_encoding import (
    ones_complement_to_twos_ir,
    signed_encoding_workspace_size,
)
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic.conversions import ones_complement_to_twos_complement

    for filename, blob in (
        (
            "ones_complement_to_twos_complement.py",
            "19a9ebd1c83a7c16161e894f7328d78a65e17bd8",
        ),
        (
            "ones_complement_to_twos_complement_test.py",
            "efbbb66c4517a79c1681c41de3a256e354d81329",
        ),
    ):
        raw = (
            Path(ones_complement_to_twos_complement.__file__).parent / filename
        ).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factory_signature_model_and_missing_source_execution() -> None:
    from qualtran import DecomposeNotImplementedError, QInt, QIntOnesComp, Side
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        _SIGNED_TO_TWOS,
        _signed_to_twos,
    )

    source = _signed_to_twos.make()
    assert _SIGNED_TO_TWOS.examples == (_signed_to_twos,)
    assert source.bitsize == 10
    assert [(r.name, r.dtype, r.side) for r in source.signature] == [
        ("x", QIntOnesComp(10), Side.LEFT),
        ("y", QInt(10), Side.RIGHT),
    ]
    assert_source_model(resources._signed_conversion_cost(10), source)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    assert source.on_classical_vals(x=-1) is NotImplemented
    with pytest.raises(NotImplementedError, match="no decomposition"):
        source.call_classically(x=-1)
    with pytest.raises(ValueError, match="through ports"):
        import_register_emitter(source)
    gates = ones_complement_to_twos_ir(
        tuple(range(10)), tuple(range(10, 18)), measurement_bit=30
    )
    assert signed_encoding_workspace_size(10) == 8
    assert sum(g.name in ("t", "tdg") for g in gates) == 32
    assert sum(g.name == "measure" for g in gates) == 8


@pytest.mark.parametrize("width", (2, 3, 5, 8, 10))
def test_original_declared_cost_is_used_in_potential_select(width: int) -> None:
    from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
        SignedIntegerToTwosComplement,
    )

    model = resources.source_select_uv_cost(width, 3, 6)
    conversions = (model.calls[0], model.calls[2])
    for call, child_width in zip(conversions, (width, width + 2), strict=True):
        assert call.repetitions == 6
        assert_source_model(call.component, SignedIntegerToTwosComplement(child_width))


def test_existing_potential_consumer_uses_replaced_conversion_without_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def replacement(width: int) -> ComponentCost:
        observed.append(width)
        return ComponentCost("conversion probe", width, GateCost(toffoli=17))

    monkeypatch.setattr(resources, "_signed_conversion_cost", replacement)
    model = resources.source_select_uv_cost(4, 3, 6)
    assert observed == [4, 6]
    assert model.calls[0].component.gates.toffoli == 17
    assert model.calls[2].component.gates.toffoli == 17
    assert model.calls[0].repetitions == model.calls[2].repetitions == 6
