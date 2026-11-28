"""Keep source logical momentum calls separate from concrete physical admission."""

from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.algorithms.first_quantization_nu_resource_model import (
    source_nu_gate_cost,
    source_nu_resource_formula,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_uniform_preparation_factories import (
    uniform_preparation_factories,
)
from flagquantum.compiler.registers.component_definition_match import _named_definitions
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._nu_preparation_library import nu_preparation_factories
from flagquantum.ecosystem._potential_preparation_library import (
    potential_preparation_factories,
)
from tests._reference_dependencies import unavailable_reference
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

try:
    import kickmix

    ordinary = import_module(
        "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu"
    )
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."
N = sympy.Symbol("n", positive=True, integer=True)


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    assert ordinary.__file__ is not None
    root = Path(ordinary.__file__).parent
    arithmetic_source = import_module(
        "qualtran.bloqs.arithmetic.multiplication"
    ).__file__
    assert arithmetic_source is not None
    arithmetic = Path(arithmetic_source).parent
    for path, blob in (
        ("multiplication.py", "efed2d6652c7dcc943c82e0c5d35ae5b3b627c09"),
        ("comparison.py", "ae1f0275d081eae99fcf7786659dd33873083f18"),
    ):
        raw = (arithmetic / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    for path, blob in (
        ("prepare_nu.py", "ae1e4e6bc279596c33c16b791ffe4d7fbe3ca788"),
        ("prepare_uv.py", "2c5e79f0c5b743a444d63e0b00048cd50c8c4060"),
        ("projectile/prepare_nu.py", "682ab3c65449b22c50edc713cbccbf7f3d3f150f"),
        ("projectile/prepare_uv.py", "f66875a9c0855bf1dab802d8ea7064bb80d017ad"),
    ):
        raw = (root / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize("electron,width", ((N, N + 1), (0, 1), (0, 0)))
def test_symbolic_six_call_roots_keep_complete_source_connections(
    projectile: bool, electron: Any, width: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        import_module(BASE + "projectile.prepare_nu").PrepareNuStateWithProj(
            electron, width, 4
        )
        if projectile
        else ordinary.PrepareNuState(width, 4)
    )
    expected = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        library = deserialize_component_library(encoded)
        with monkeypatch.context() as patch:
            import sys

            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            record = resolve_component_program(library, 0, nu_preparation_factories())
        assert verify_program_record(record) == ()
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert _named_inputs(_native_trace(record)) == expected
        assert len(record.functions[0].blocks[0].operations) == 6
        assert sum(not f.blocks for f in record.functions) == 5
        if baseline is not None:
            assert record == baseline
        baseline = record
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(record, (), clean=tuple(range(100)))


@pytest.mark.parametrize("electron,width", ((2, N), (N, 3), (3, 2)))
def test_uv_parent_survives_original_relational_child_failure(
    electron: Any, width: Any
) -> None:
    source = import_module(
        BASE + "projectile.prepare_uv"
    ).PrepareUVFirstQuantizationWithProj(electron, width, N, 3, 4, N, N + 2)
    nu = import_module(BASE + "projectile.prepare_nu").PrepareNuStateWithProj(
        electron, width, 4
    )
    with pytest.raises((TypeError, ValueError)):
        nu.decompose_bloq()
    header = resolve_component_program(_library(nu, 0), 0, nu_preparation_factories())
    assert not header.functions[0].blocks
    expected = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    for depth in (0, 1):
        if (electron, width) == (3, 2) and depth == 1:
            # Depth zero omits complexity annotations; deeper serialization
            # queries a model that tries constructing the invalid mu child.
            with pytest.raises(ValueError, match="bitsize_n < bitsize_p"):
                _library(source, depth)
            continue
        library = _library(source, depth)
        record = resolve_component_program(
            library, 0, potential_preparation_factories()
        )
        assert verify_program_record(record) == ()
        assert _named_inputs(_native_trace(record)) == expected
        assert len(record.functions[0].blocks[0].operations) == 2
        assert sum(not f.blocks for f in record.functions) == 2


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize(
    "kind",
    (
        "PrepareNuSuperPositionState",
        "FlagZeroAsFailure",
        "TestNuLessThanMu",
        "TestNuInequality",
    ),
)
def test_opaque_symbolic_leaves_retain_width_and_flag_direction(
    kind: str, adjoint: bool
) -> None:
    args = (N + 1, N) if kind == "TestNuInequality" else (N + 1,)
    source = getattr(ordinary, kind)(*args, is_adjoint=adjoint)
    for depth in (0, 1, 2):
        library = _library(source, depth)
        record = resolve_component_program(library, 0, nu_preparation_factories())
        assert verify_program_record(record) == ()
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert not record.functions[0].blocks


@pytest.mark.parametrize(
    "width,sample", ((2, 3), (N + 1, 4), (0, 4), (sympy.Integer(3), 4))
)
def test_ordinary_nu_exact_logical_inverse_reverses_flags_and_uniform_child(
    width: Any, sample: int
) -> None:
    source = ordinary.PrepareNuState(width, sample).adjoint()
    factories = nu_preparation_factories()
    forward = factories[BASE + "prepare_nu.PrepareNuState"](
        "forward", None, {"num_bits_p": width, "m_param": sample}
    )
    assert isinstance(forward, ProgramRecord)
    direct = factories["qualtran._infra.adjoint.Adjoint"](
        "main", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(direct, ProgramRecord)
    assert _named_inputs(_native_trace(direct)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    for depth in (0, 1):
        library = _library(source, depth)
        record = resolve_component_program(library, 0, nu_preparation_factories())
        if isinstance(width, sympy.Integer):
            # The original argument codec turns this literal into Python int.
            # That changes the source greedy order. Keep both source paths.
            encoded_width = next(
                arg.value
                for entry in library.entries
                if entry.declaration.name == BASE + "prepare_nu.PrepareNuState"
                for arg in entry.declaration.arguments
                if arg.name == "num_bits_p"
            )
            assert type(encoded_width) is int
            expected = ordinary.PrepareNuState(encoded_width, sample).adjoint()
        else:
            expected = source
        assert verify_program_record(record) == ()
        assert _named_inputs(_native_trace(record)) == _named_inputs(
            _source_trace(SourceCalls, expected.decompose_bloq())
        )
        assert len(record.functions[0].blocks[0].operations) == 6


@pytest.mark.parametrize("projectile", (False, True))
def test_exact_potential_outer_inverse_keeps_original_opaque_child_boundary(
    projectile: bool,
) -> None:
    module = import_module(
        BASE + ("projectile.prepare_uv" if projectile else "prepare_uv")
    )
    fields = dict(
        num_bits_p=1, eta=2, num_atoms=3, m_param=4, lambda_zeta=3, num_bits_nuc_pos=8
    )
    if projectile:
        fields["num_bits_n"] = 2
    source = getattr(
        module,
        (
            "PrepareUVFirstQuantizationWithProj"
            if projectile
            else "PrepareUVFirstQuantization"
        ),
    )(**fields).adjoint()
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, potential_preparation_factories()
        )
        assert verify_program_record(record) == ()
        assert _named_inputs(_native_trace(record)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
        assert len(record.functions[0].blocks[0].operations) == 2
        if projectile:
            reachable = _named_definitions(record, record.functions[0].name)
            assert sum(not f.blocks for f in reachable) == 2


@pytest.mark.parametrize("sample", (-1, 0, 1))
def test_source_failed_sample_body_retains_only_its_logical_header(sample: int) -> None:
    source = ordinary.PrepareNuState(2, sample)
    with pytest.raises((ValueError, import_module("qualtran").BloqError)):
        source.decompose_bloq()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, nu_preparation_factories())
    assert verify_program_record(record) == ()
    assert not record.functions[0].blocks
    assert record.functions[0].registers == library.entries[0].declaration.registers


@pytest.mark.parametrize("width", (0, 1, 2, 3, sympy.Integer(3), N + 1, 2 * N + 1))
@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_source_declared_categories_and_peak_are_separate_from_execution(
    width: Any, projectile: bool, adjoint: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = (
        import_module(BASE + "projectile.prepare_nu").PrepareNuStateWithProj(
            0, width, 4
        )
        if projectile
        else ordinary.PrepareNuState(width, 4)
    )
    directed = source.adjoint() if adjoint else source
    fields = {"num_bits_n": width} if projectile else {}
    expected = get_cost_value(directed, QECGatesCost()).asdict()
    actual = source_nu_gate_cost(
        0 if projectile else width, 4, adjoint=adjoint, **fields
    ).asdict()
    assert actual.keys() == expected.keys()
    for key in actual:
        assert sympy.simplify(actual[key] - expected[key]) == 0
    if projectile and adjoint:
        with pytest.raises(ValueError, match="no valid peak"):
            source_nu_resource_formula(0, 4, num_bits_n=width, adjoint=True)
        return
    result = source_nu_resource_formula(
        0 if projectile else width, 4, adjoint=adjoint, **fields
    )
    assert (
        sympy.simplify(result.logical_qubits - get_cost_value(directed, QubitCount()))
        == 0
    )


@pytest.mark.parametrize("width,atoms,charge", ((0, 0, 3), (2, 1, 3), (N + 1, 3, N)))
@pytest.mark.parametrize("projectile", (False, True))
def test_uv_symbolic_categories_and_disjoint_live_bystanders(
    width: Any, atoms: int, charge: Any, projectile: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.first_quantization_potential_formula import (
        source_uv_gate_cost,
        source_uv_resource_formula,
    )

    module = import_module(
        BASE + ("projectile.prepare_uv" if projectile else "prepare_uv")
    )
    fields = dict(
        num_bits_p=0 if projectile else width,
        eta=N,
        num_atoms=atoms,
        m_param=4,
        lambda_zeta=charge,
        num_bits_nuc_pos=N,
    )
    if projectile:
        fields["num_bits_n"] = width
    original = getattr(
        module,
        (
            "PrepareUVFirstQuantizationWithProj"
            if projectile
            else "PrepareUVFirstQuantization"
        ),
    )(**fields)
    options = {"num_bits_n": width} if projectile else {}
    gates = source_uv_gate_cost(0 if projectile else width, 4, charge, **options)
    expected = get_cost_value(original, QECGatesCost()).asdict()
    assert gates.asdict().keys() == expected.keys()
    for key, count in gates.asdict().items():
        assert sympy.simplify(count - expected[key]) == 0
    result = source_uv_resource_formula(
        0 if projectile else width, atoms, 4, charge, **options
    )
    assert (
        sympy.simplify(result.logical_qubits - get_cost_value(original, QubitCount()))
        == 0
    )


@pytest.mark.parametrize("sample", (-1, 0, 1))
def test_failed_nu_sample_keeps_its_independent_source_gate_declaration(
    sample: int,
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = ordinary.PrepareNuState(2, sample)
    assert (
        source_nu_gate_cost(2, sample).asdict()
        == get_cost_value(original, QECGatesCost()).asdict()
    )
    with pytest.raises(ValueError, match="no valid peak"):
        source_nu_resource_formula(2, sample)


@pytest.mark.parametrize(
    "left,right", ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (0, 6), (6, 0))
)
@pytest.mark.parametrize("legacy", (False, True))
def test_zero_word_comparison_categories_are_not_a_physical_tree(
    left: int, right: int, legacy: bool
) -> None:
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.comparison.comparison_resource_model import (
        source_less_equal_gate_cost,
    )

    expected = get_cost_value(
        LessThanEqual(left, right), QECGatesCost(legacy_shims=legacy)
    )
    assert (
        source_less_equal_gate_cost(left, right, legacy_shims=legacy).asdict()
        == expected.asdict()
    )


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("electron,width", ((0, 0), (0, 1), (N, N + 1), (1, N + 1)))
def test_projectile_mu_symbolic_model_retains_source_flag_and_categories(
    electron: Any, width: Any, adjoint: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.first_quantization_nu_resource_model import (
        source_projectile_prepare_mu_formula,
    )

    source = import_module(
        BASE + "projectile.prepare_nu"
    ).PrepareMuUnaryEncodedOneHotWithProj(width, electron, adjoint)
    result = source_projectile_prepare_mu_formula(electron, width, adjoint=adjoint)
    assert (
        sympy.simplify(result.toffoli - get_cost_value(source, QECGatesCost()).toffoli)
        == 0
    )
    assert (
        sympy.simplify(result.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )
    assert source.signature.get_right("flag").side.name == "RIGHT"


def test_codec_lost_positivity_never_becomes_an_inferred_executable_root() -> None:
    module = import_module(BASE + "projectile.prepare_nu")
    original = module.PrepareNuStateWithProj(1, N + 1, 4)
    expected = _named_inputs(_source_trace(SourceCalls, original.decompose_bloq()))
    constructor = BASE + "projectile.prepare_nu.PrepareNuStateWithProj"
    direct = nu_preparation_factories()[constructor](
        "main", None, {"num_bits_p": 1, "num_bits_n": N + 1, "m_param": 4}
    )
    assert isinstance(direct, ProgramRecord)
    assert _named_inputs(_native_trace(direct)) == expected
    for depth in (0, 1):
        library = _library(original, depth)
        fields = {
            argument.name: argument.value
            for argument in library.entries[0].declaration.arguments
        }
        wire_width = fields["num_bits_n"]
        assert isinstance(wire_width, sympy.Expr)
        assert N not in wire_width.free_symbols
        with pytest.raises(TypeError, match="truth value"):
            module.PrepareNuStateWithProj(**fields).decompose_bloq()
        named = resolve_component_program(library, 0, nu_preparation_factories())
        assert not named.functions[0].blocks
        if depth:
            # Stored edges retain more information than failed constructor
            # reconstruction. The existing typed graph reader needs no guess.
            factories = {
                **uniform_preparation_factories(),
                **nu_preparation_factories(),
            }
            del factories[constructor]
            stored = resolve_component_program(library, 0, factories)
            assert verify_program_record(stored) == ()
            assert _named_inputs(_native_trace(stored)) == expected


def test_unused_potential_metadata_does_not_gain_declaration_constraints() -> None:
    source = import_module(
        BASE + "projectile.prepare_uv"
    ).PrepareUVFirstQuantizationWithProj(
        1, 2, "unused eta", 3, 4, 3, "unused precision"
    )
    for depth in (0, 1):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        restored = deserialize_component_library(encoded)
        assert serialize_component_library(restored) == encoded
        record = resolve_component_program(
            restored, 0, potential_preparation_factories()
        )
        assert _named_inputs(_native_trace(record)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
    zeta = import_module(BASE + "prepare_zeta").PrepareZetaState(
        3, "invalid cost", "unused precision"
    )
    library = _library(zeta, 0)
    record = resolve_component_program(library, 0, potential_preparation_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert not record.functions[0].blocks
