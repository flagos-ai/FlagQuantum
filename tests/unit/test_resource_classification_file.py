"""Match the frozen classifier's default fixture, precedence and vector predicate."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.resource_classification import (
    attribute_t_cost,
    classify_resource_family,
    is_resource_state_or_effect,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ArithmeticCall,
    CliffordCall,
    ConjunctionCall,
    ControlledSwapCall,
    FixedTCall,
    MeasurementCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    identity_cost_key,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.compiler.registers.comparison.less_than import (
    source_less_than_constant_cost,
)
from flagquantum.compiler.registers.lookup.qrom_resource_model import source_qrom_cost
from flagquantum.compiler.registers.phase.hamming_weight_phasing import (
    source_hamming_weight_phasing_cost,
)
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    source_prepare_reflection_cost,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from flagquantum.compiler.registers.primitives.basis_state import (
    basis_effect_cost_key,
    basis_state_cost_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_classification_source_and_suite() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/resource_counting"
    )
    for file, expected in (
        ("classify_bloqs.py", "53e5c97acf2ec9350d4c04fde31473f7307f4bdf"),
        ("classify_bloqs_test.py", "753a8c2c3b8e079d83a9cfee21e01aee1b51c8e3"),
    ):
        path = root / file
        if not path.is_file():
            pytest.skip("Frozen classification files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_basis_vector_predicate_matches_all_original_vector_kinds() -> None:
    source = pytest.importorskip("qualtran.resource_counting.classify_bloqs")
    from qualtran.bloqs.basic_gates import (
        Identity,
        MinusEffect,
        MinusState,
        OneEffect,
        OneState,
        PlusEffect,
        PlusState,
        XGate,
        ZeroEffect,
        ZeroState,
    )

    cases: tuple[tuple[ResourceCall, object], ...] = (
        (basis_state_cost_key("x", 0), PlusState()),
        (basis_state_cost_key("x", 1), MinusState()),
        (basis_effect_cost_key("x", 0), PlusEffect()),
        (basis_effect_cost_key("x", 1), MinusEffect()),
        (basis_state_cost_key("z", 0), ZeroState()),
        (basis_state_cost_key("z", 1), OneState()),
        (basis_effect_cost_key("z", 0), ZeroEffect()),
        (basis_effect_cost_key("z", 1), OneEffect()),
        (identity_cost_key(1), Identity()),
        (CliffordCall("x", 1), XGate()),
    )
    for index, (native, upstream) in enumerate(cases):
        assert (
            is_resource_state_or_effect(native)
            == source.bloq_is_state_or_effect(upstream)
            == (index < 8)
        )
    assert not is_resource_state_or_effect(MeasurementCall("z"))


def test_typed_defaults_and_custom_first_match_precedence() -> None:
    source = pytest.importorskip("qualtran.resource_counting.classify_bloqs")
    from qualtran import QInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant
    from qualtran.bloqs.basic_gates import CSwap, Rz, TGate, Toffoli
    from qualtran.bloqs.mcmt.and_bloq import And

    pairs = (
        (ArithmeticCall("add", (8, 8)), Add(QInt(8))),
        (
            ArithmeticCall("less_than", (8,), adjoint=True),
            LessThanConstant(8, 3).adjoint(),
        ),
        (ControlledSwapCall(10), CSwap(10)),
        (RotationCall("rz", -0.3), Rz(-0.3)),
        (ConjunctionCall((0, 1)), And(0, 1)),
        (PhaseCall("t", adjoint=True), TGate().adjoint()),
        (FixedTCall("toffoli"), Toffoli()),
    )
    for native, upstream in pairs:
        assert classify_resource_family(native) == source.classify_bloq(
            upstream, source._get_basic_bloq_classification()
        )
    swap = ControlledSwapCall(10)

    def matches_all(_node: ResourceCall) -> bool:
        return True

    def matches_swap(node: ResourceCall) -> bool:
        return isinstance(node, ControlledSwapCall)

    rules = ((matches_all, "first"), (matches_swap, "swaps"))
    upstream_broad = "qualtran.bloqs"
    upstream_swap = "qualtran.bloqs.basic_gates.swap"
    assert classify_resource_family(swap, rules) == source.classify_bloq(
        CSwap(10), {upstream_broad: "first", upstream_swap: "swaps"}
    )
    assert classify_resource_family(swap, rules[::-1]) == source.classify_bloq(
        CSwap(10), {upstream_swap: "swaps", upstream_broad: "first"}
    )
    assert classify_resource_family(swap, ()) == "other"


def test_all_original_default_family_bins_use_native_cost_models() -> None:
    source = pytest.importorskip("qualtran.resource_counting.classify_bloqs")
    cases = pytest.importorskip("qualtran.resource_counting.classify_bloqs_test")
    from qualtran import QInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    rows = (4, 10, 11, 34)
    identity = ComponentCost("PrepareIdentity", 8)
    fixture = (
        (ComponentCost("ModelSwap", 21, GateCost(cswap=10)), CSwap(10), 42, "swaps"),
        (
            source_hamming_weight_phasing_cost(10, 1.11),
            HammingWeightPhasing(10, 1.11),
            11,
            "rotations",
        ),
        (source_add_cost(8, 8), Add(QInt(8)), 4, "arithmetic"),
        (
            source_qrom_cost(0, (2,), (4,), ((1, 6),), (rows,)),
            QROM.build_from_data(rows),
            8,
            "data_loading",
        ),
        (
            ComponentCost("ModelAnd", 3, GateCost(and_bloq=1)),
            And(),
            4,
            "multi_control_pauli",
        ),
        (
            source_prepare_reflection_cost(8, 0, identity, identity, controlled=False),
            ReflectionUsingPrepare(PrepareIdentity.from_bitsizes((3, 3, 2))),
            100,
            "reflection",
        ),
        (
            source_less_than_constant_cost(8, 3),
            LessThanConstant(8, 3),
            10,
            "arithmetic",
        ),
        (
            source_uniform_preparation_cost(4),
            PrepareUniformSuperposition(4),
            2,
            "state_preparation",
        ),
    )
    leaf_families: dict[ResourceCall, str] = {
        node: label for node, _, _, label in fixture
    }

    root = ComponentCost(
        "root", 21, calls=tuple(CostCall(node, count) for node, _, count, _ in fixture)
    )
    expected = source.classify_t_count_by_bloq_type(
        cases.TestBundleOfBloqs(
            tuple((upstream, count) for _, upstream, count, _ in fixture)
        )
    )
    assert attribute_t_cost(root, leaf_families.get) == expected
    assert "state_preparation" not in expected
