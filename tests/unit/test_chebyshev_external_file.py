"""Exact external source recipes needed by complete Chebyshev block binding."""

from collections import Counter
from collections.abc import Callable, Mapping
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_atomic_source_fredkin_uses_its_literal_clifford_t_recipe() -> None:
    import cirq
    from qualtran.bloqs.basic_gates import TwoBitCSwap
    from qualtran.bloqs.basic_gates import swap as source_module
    from qualtran.bloqs.block_encoding import Unitary

    raw = Path(source_module.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "5c2481f7a1ee0a592d1192999ddf7e6dba8facc3"
    )
    leaf = TwoBitCSwap()
    named = tuple(map(cirq.NamedQubit, ("ctrl", "x", "y")))
    exact = import_cirq_feedback(
        cirq.Circuit(leaf.to_clifford_t_circuit()), qubit_order=named
    ).ir
    block = import_block_encoding(Unitary(leaf), include_controlled=False)
    for adjoint in (False, True):
        # The source leaf is self-adjoint and exposes the same literal recipe.
        operation = block.make_on(system=(0, 1, 2), adjoint=adjoint)
        ir = Circuit(3).append(operation).to_ir()
        assert ir.instructions == exact.instructions
        counts = Counter(g.name for g in ir.instructions)
        assert counts["t"] + counts["tdg"] == 7
        assert len(ir.instructions) == 17
        for actual in (ir, CircuitIR.from_json(ir.to_json())):
            torch.testing.assert_close(
                get_unitary(actual, dtype=torch.complex128),
                torch.as_tensor(leaf.tensor_contract(), dtype=torch.complex128),
                atol=3e-12,
                rtol=0,
            )


@pytest.mark.parametrize("adjoint", (False, True))
def test_original_controlled_comparator_has_no_partial_register_decomposition(
    adjoint: bool,
) -> None:
    import cirq
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.arithmetic.comparison import SingleQubitCompare

    leaf = SingleQubitCompare(is_adjoint=adjoint).controlled()
    with pytest.raises(DecomposeTypeError, match="non-thru registers"):
        leaf.decompose_bloq()
    assert cirq.decompose_once(leaf.on(*cirq.LineQubit.range(5)), default=None) is None


@pytest.fixture(scope="module", autouse=True)
def frozen_chebyshev_sources() -> None:
    from importlib.metadata import version

    import kickmix
    from qualtran.bloqs.block_encoding import chebyshev_polynomial

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(chebyshev_polynomial.__file__).parent
    for name, blob in (
        ("chebyshev_polynomial.py", "327d0ace4b411c9d8290d59d496e717b0c9e3915"),
        ("chebyshev_polynomial_test.py", "2bc29ea862924b8e213d94b9f3f15ff1c53414b9"),
        ("chebyshev_polynomial.ipynb", "30fa0a9688a8704c7ed9cc32619ca7ed96c93ada"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob


@pytest.mark.parametrize("odd", (False, True))
def test_actual_original_child_replaces_chebyshev_consumer(odd: bool) -> None:
    from itertools import product

    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _chebyshev_poly_even,
        _chebyshev_poly_odd,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
        ChebyshevBlockEncoding,
    )
    from flagquantum.circuit import _CircuitOperation
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    source = (_chebyshev_poly_odd if odd else _chebyshev_poly_even)()
    external = import_block_encoding(source.block_encoding, include_controlled=False)
    native = ChebyshevBlockEncoding(external, source.order)
    child_work, child_slots = external.requirements()
    reflection_work, reflection_slots = native.reflection_requirements()
    logical = native.system_bitsize + native.ancilla_bitsize + native.resource_bitsize
    work = tuple(range(logical, logical + max(child_work, reflection_work)))
    grid = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(grid + 1, grid % 5 - 2)
    state /= state.norm()
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        scratch: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return external.make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            workspace=scratch,
            classical_bits=slots,
            adjoint=inverse,
        )

    costs = tuple(
        ComponentCost(
            "Independent original child",
            int(get_cost_value(child, QubitCount())),
            GateCost(**get_cost_value(child, QECGatesCost()).asdict()),
        )
        for child in (source.block_encoding, source.block_encoding.adjoint())
    )
    for adjoint in (False, True):
        operation = native.make_on(
            bind,
            system=tuple(range(native.system_bitsize)),
            ancilla=tuple(
                range(native.system_bitsize, logical - native.resource_bitsize)
            ),
            resource=tuple(range(logical - native.resource_bitsize, logical)),
            workspace=work[:child_work],
            classical_bits=tuple(range(child_slots)),
            reflection_workspace=work[:reflection_work],
            reflection_classical_bits=tuple(range(reflection_slots)),
            adjoint=adjoint,
        )
        ir = Circuit(logical + len(work)).append(operation).to_ir()
        locked = source.adjoint() if adjoint else source
        assert_source_model(
            native.resource_model(costs[0], costs[1], adjoint=adjoint), locked
        )
        expected = torch.kron(
            torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128) @ state,
            clean,
        )
        count = sum(g.name == "measure" for g in ir.instructions)
        assert count == (0 if odd else 8)
        for history in product((0, 1), repeat=count):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), ir.instructions, history),
                expected * 2 ** (-count / 2),
                atol=8e-12,
                rtol=0,
            )
        restored = CircuitIR.from_json(ir.to_json())
        assert restored.instructions == ir.instructions


@pytest.mark.parametrize("order", (0, 1, 2, 3))
@pytest.mark.parametrize("scaled", (False, True))
def test_symbolic_packed_declaration_resolves_to_actual_external_children(
    order: int, scaled: bool
) -> None:

    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.bloqs.block_encoding import (
        ChebyshevPolynomial,
        ScaledChebyshevPolynomial,
        Unitary,
    )
    from sympy import Symbol, sympify

    from flagquantum.algorithms.primitives._block_metadata import (
        _packed_block_signature,
    )
    from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
        ChebyshevBlockEncoding,
    )
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
        source_linear_combination_signature,
    )
    from flagquantum.algorithms.primitives.scaled_chebyshev_block_encoding import (
        ScaledChebyshevBlockEncoding,
    )
    from flagquantum.circuit import _CircuitOperation
    from tests.unit.measurement_certificate import certify_erasure_histories
    from tests.unit.test_scaled_chebyshev_block_encoding import _bind_lcu, _lcu_scratch

    s, a, r = (Symbol(name) for name in ("s", "a", "r"))
    source_cls = ScaledChebyshevPolynomial if scaled else ChebyshevPolynomial
    symbolic_child = Unitary(
        Identity(s), alpha=1.5, ancilla_bitsize=a, resource_bitsize=r
    )
    declaration = source_cls(symbolic_child, order)
    # The source's explicit guard is on symbolic signal width. No physical
    # source or native circuit is claimed for an unresolved declaration.
    guarded = declaration.linear_combination if scaled else declaration
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        guarded.decompose_bloq()
    substitutions = {s: 1, a: 1, r: 1}
    child = Unitary(Identity(1), alpha=1.5, ancilla_bitsize=1, resource_bitsize=1)
    source = source_cls(child, order)
    external = import_block_encoding(child)
    native = (
        ScaledChebyshevBlockEncoding(external, order)
        if scaled
        else ChebyshevBlockEncoding(external, order)
    )
    lowered = native.linear_combination if scaled else native
    if isinstance(lowered, LinearCombinationBlockEncoding):
        symbolic_signature = source_linear_combination_signature(
            s,
            (a,) * (order + 1),
            (r,) * (order + 1),
            prepare_selection=lowered.prepare.selection_width,
            prepare_junk=lowered.prepare.junk_width,
        )
    else:
        symbolic_signature = _packed_block_signature(s, a, r)
    assert [(reg.name, sympify(reg.bitsize)) for reg in symbolic_signature] == [
        (reg.name, sympify(reg.bitsize)) for reg in declaration.signature
    ]
    assert [(reg.name, reg.bitsize) for reg in native.signature] == [
        (reg.name, int(sympify(reg.bitsize).subs(substitutions)))
        for reg in declaration.signature
    ]
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    logical = native.system_bitsize + native.ancilla_bitsize + native.resource_bitsize
    ports = dict(
        system=(0,),
        ancilla=tuple(range(1, 1 + native.ancilla_bitsize)),
        resource=tuple(range(1 + native.ancilla_bitsize, logical)),
    )

    def plain(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return external.make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            workspace=work,
            classical_bits=slots,
            adjoint=inverse,
        )

    def controlled_term(
        term: ChebyshevBlockEncoding,
    ) -> Callable[
        [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        _CircuitOperation,
    ]:
        def bind(
            predicate: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            inverse: bool,
        ) -> _CircuitOperation:
            def child(
                ports: Mapping[str, tuple[int, ...]],
                work: tuple[int, ...],
                slots: tuple[int, ...],
                child_inverse: bool,
            ) -> _CircuitOperation:
                return external.bind_child(predicate, ports, work, slots, child_inverse)

            return term.make_on(
                child,
                system=ports["system"],
                ancilla=ports["ancilla"],
                resource=ports["resource"],
                workspace=work,
                classical_bits=slots,
                control=predicate,
                adjoint=inverse,
            )

        return bind

    for adjoint in (False, True):
        if isinstance(lowered, ChebyshevBlockEncoding):
            nw, nb = lowered.reflection_requirements()
            work = tuple(range(logical, logical + nw))
            operation = lowered.make_on(
                plain,
                **ports,
                reflection_workspace=work,
                reflection_classical_bits=tuple(range(nb)),
                adjoint=adjoint,
            )
        else:
            scratch = _lcu_scratch(lowered, False)
            work = tuple(range(logical, logical + scratch))
            operation = _bind_lcu(
                lowered,
                tuple(controlled_term(term) for term in lowered.block_encodings),
                ports,
                work,
                tuple(range(scratch)),
                adjoint=adjoint,
            )
        ir = Circuit(logical + len(work)).append(operation).to_ir()
        # Actual numeric binding follows explicit resolution and keeps all
        # logical outputs. Only private work is required to return to zero.
        certify_erasure_histories(ir.instructions, n_wires=ir.n_wires, zero_wires=work)
        assert CircuitIR.from_json(ir.to_json()).instructions == ir.instructions


@pytest.mark.parametrize("order", (0, 1, 2, 3, 5))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("legacy", (False, True))
def test_symbolic_chebyshev_fallback_counts_and_zero_count_callee_peaks(
    order: int, adjoint: bool, legacy: bool
) -> None:
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding import ChebyshevPolynomial, Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol, simplify

    from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
        source_symbolic_chebyshev_resource_formula,
    )
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        source_unitary_block_resource_formula,
    )
    from flagquantum.compiler._component_trace.resource_model import GateCost
    from flagquantum.compiler.registers.preparation.prepare_reflection import (
        source_symbolic_zero_reflection_resource_formula,
    )
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    a, r = Symbol("a"), Symbol("r")
    source = ChebyshevPolynomial(
        Unitary(TGate(), ancilla_bitsize=a, resource_bitsize=r), order
    )
    reflected = source.reflection_bloq
    reflection = source_symbolic_zero_reflection_resource_formula(
        a, legacy_shims=legacy
    )
    rcost = get_cost_value(reflected, QECGatesCost(legacy_shims=legacy))
    for name in GateCost.__dataclass_fields__:
        assert simplify(getattr(reflection, name) - getattr(rcost, name)) == 0
    assert (
        simplify(reflection.logical_qubits - get_cost_value(reflected, QubitCount()))
        == 0
    )
    child = source_unitary_block_resource_formula(
        SymbolicResourceFormula(0, 0, 0, 1, t=1), ancilla=a, resource=r
    )
    model = source_symbolic_chebyshev_resource_formula(
        1, a, r, order, child, child, adjoint=adjoint, legacy_shims=legacy
    )
    locked = source.adjoint() if adjoint else source
    expected = get_cost_value(locked, QECGatesCost(legacy_shims=legacy))
    for name in GateCost.__dataclass_fields__:
        assert simplify(getattr(model, name) - getattr(expected, name)) == 0
    assert simplify(model.logical_qubits - get_cost_value(locked, QubitCount())) == 0


def test_symbolic_models_do_not_admit_resolved_or_invalid_queries() -> None:
    from sympy import Symbol

    from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
        source_symbolic_chebyshev_resource_formula,
    )
    from flagquantum.compiler.registers.preparation.prepare_reflection import (
        source_symbolic_zero_reflection_resource_formula,
    )
    from flagquantum.compiler.registers.resources.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    with pytest.raises(ValueError, match="concrete reflection"):
        source_symbolic_zero_reflection_resource_formula(3)
    child = SymbolicResourceFormula(0, 0, 0, 1)
    with pytest.raises(ValueError, match="unresolved signal"):
        source_symbolic_chebyshev_resource_formula(1, 1, 0, 2, child, child)
    with pytest.raises(ValueError, match="nonnegative order"):
        source_symbolic_chebyshev_resource_formula(1, Symbol("a"), 0, -1, child, child)


def test_all_original_factory_serialization_outcomes_and_capture_admission() -> None:
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _chebyshev_poly_even,
        _chebyshev_poly_odd,
        _scaled_chebyshev_poly_even,
        _scaled_chebyshev_poly_odd,
    )
    from qualtran.testing import BloqCheckResult, check_bloq_example_serializes

    from tests.unit.measurement_certificate import certify_erasure_histories

    for example, diagnostic, physical in (
        (_chebyshev_poly_even, "SpecializeOnCtrlBit.ONE", (8, 520)),
        (_chebyshev_poly_odd, "", (1, 5)),
        (_scaled_chebyshev_poly_even, "Unable to find a Bloq", None),
        (_scaled_chebyshev_poly_odd, "Unable to find a Bloq", (32, 989)),
    ):
        outcome, details = check_bloq_example_serializes(example)
        assert outcome is (BloqCheckResult.FAIL if diagnostic else BloqCheckResult.PASS)
        assert diagnostic in details
        if physical is None:
            with pytest.raises(ValueError, match="SingleQubitCompare"):
                import_block_encoding(example(), include_controlled=False)
        else:
            block = import_block_encoding(example(), include_controlled=False)
            logical = (
                block.system_bitsize + block.ancilla_bitsize + block.resource_bitsize
            )
            for ir in block.snapshots[:2]:
                assert (ir.n_wires, len(ir.instructions)) == physical
                assert CircuitIR.from_json(ir.to_json()).to_json() == ir.to_json()
                events = certify_erasure_histories(
                    ir.instructions,
                    n_wires=ir.n_wires,
                    zero_wires=tuple(range(logical, ir.n_wires)),
                )
                assert events == sum(g.name == "measure" for g in ir.instructions)


def test_actual_scaled_odd_root_matches_native_full_sparse_outputs() -> None:
    from math import sqrt

    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _scaled_chebyshev_poly_odd,
    )

    from tests.unit.test_scaled_chebyshev_block_encoding import (
        original_scaled_operation,
    )
    from tests.unit.test_sparse_chemistry_prepare import _branch

    source = _scaled_chebyshev_poly_odd()
    external = import_block_encoding(source, include_controlled=False)
    logical = (
        external.system_bitsize + external.ancilla_bitsize + external.resource_bitsize
    )
    work, slots = external.requirements()
    size = logical + work
    assert (logical, work, slots, size) == (18, 14, 34, 32)
    for adjoint in (False, True):
        scaled, operation = original_scaled_operation(5, adjoint=adjoint)
        assert scaled.signature.n_qubits() == logical
        native = Circuit(size).append(operation).to_ir().instructions
        foreign = external.make_on(
            system=(0,),
            ancilla=tuple(range(1, 1 + external.ancilla_bitsize)),
            resource=tuple(range(1 + external.ancilla_bitsize, logical)),
            workspace=tuple(range(logical, size)),
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        ).instructions
        counts = Counter(g.name for g in foreign)
        assert counts == Counter(g.name for g in native)
        assert len(foreign) == len(native) == 989
        assert counts["t"] + counts["tdg"] == 178 and counts["measure"] == 34
        initial = {0: (1 + 2j) / sqrt(10), 1 << (size - 1): (2 - 1j) / sqrt(10)}
        for history in ((0,) * 34, tuple(i % 2 for i in range(34))):
            actual = _branch(
                foreign, size, history, initial, normalize_measurements=True
            )
            expected = _branch(
                native, size, history, initial, normalize_measurements=True
            )
            assert actual == pytest.approx(expected, abs=5e-10)
            assert all(index & ((1 << work) - 1) == 0 for index in actual)


@pytest.mark.parametrize("width", (0, 1, 2, 3))
def test_symbolic_integer_reflection_diagnostics_match_source_admission(
    width: int,
) -> None:
    from qualtran import BloqError
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Integer

    from flagquantum.compiler._component_trace.resource_model import GateCost
    from flagquantum.compiler.registers.preparation.prepare_reflection import (
        source_symbolic_zero_reflection_resource_formula,
    )

    declaration = Integer(width)
    source = ReflectionUsingPrepare.reflection_around_zero(
        bitsizes=(declaration,), global_phase=-1
    )
    if width < 2:
        exception, diagnostic = (
            (BloqError, "does not accept Soquets")
            if width == 0
            else (ValueError, "symbolic CtrlSpec")
        )
        with pytest.raises(exception, match=diagnostic):
            get_cost_value(source, QECGatesCost())
        with pytest.raises(ValueError, match="zero/one reflection"):
            source_symbolic_zero_reflection_resource_formula(declaration)
    else:
        model = source_symbolic_zero_reflection_resource_formula(declaration)
        expected = get_cost_value(source, QECGatesCost())
        assert all(
            getattr(model, name) == getattr(expected, name)
            for name in GateCost.__dataclass_fields__
        )
        assert model.logical_qubits == get_cost_value(source, QubitCount())
