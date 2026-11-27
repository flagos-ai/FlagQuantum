"""Compare independent guided Hamiltonian models with the locked declarations."""

from __future__ import annotations

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.guided_hamiltonian_models import (
    symbolic_guided_hamiltonian,
    symbolic_guided_phase_estimation,
)
from flagquantum.algorithms.primitives.guided_hamiltonian_resource_model import (
    GuidedHamiltonianModel,
    GuidedPhaseEstimationModel,
)
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

_PHASE = "GuidedHamiltonianPhaseEstimation"
_ROOT = "GuidedHamiltonian"
_QPE = "QubitizationQPE"


def _widths(original: Any) -> tuple[Any, Any, Any, Any]:
    return (
        original.hamiltonian.system_bitsize,
        original.hamiltonian.ancilla_bitsize,
        original.guiding_state.selection_bitsize,
        original.guiding_state.junk_bitsize,
    )


def _assert_signature(signature: Signature, original: Any) -> None:
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in original.signature
    ]
    assert signature[0].dtype.encoding == "fixed_point"
    assert dict(signature[0].dtype.parameters) == {
        "fractional_width": original.signature[0].dtype.num_frac,
        "signed": False,
    }
    assert all(r.dtype.encoding == "opaque" for r in tuple(signature)[1:])


def _qpe_keys(qpe: Any, *, adjoint: bool = False) -> dict[Any, str]:
    from qualtran._infra.controlled import CtrlSpec

    calls = {
        qpe.ctrl_state_prep: "KaiserWindowState",
        qpe.walk.controlled(): "ControlledWalk",
        qpe.walk.reflect.controlled(
            ctrl_spec=CtrlSpec(cvs=0)
        ): "ZeroControlledReflection",
        qpe.walk: "Walk",
        qpe.qft_inv: "QFTWithReverse†",
    }
    if not adjoint:
        return calls
    return {
        node.adjoint(): {
            "ZeroControlledReflection": "ZeroControlledReflection",
            "QFTWithReverse†": "QFTWithReverse",
        }.get(name, f"{name}†")
        for node, name in calls.items()
    }


def test_locked_guided_package_and_symbolic_roots() -> None:
    import kickmix
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran._infra.controlled import CtrlSpec
    from qualtran.bloqs import hamiltonian_simulation as package
    from qualtran.bloqs.hamiltonian_simulation import guided_hamiltonian as source
    from qualtran.resource_counting._call_graph import get_bloq_callee_counts

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    for module, digest in (
        (package, "20d19a507dc9f99ef148435bf1af2e8f3a8f9bff"),
        (source, "847914ba4e8437b6bf0affab4262beb89c5b0a15"),
    ):
        assert module.__file__ is not None
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    assert getattr(package, "GuidedHamiltonian") is source.GuidedHamiltonian
    assert (
        getattr(package, "GuidedHamiltonianPhaseEstimation")
        is source.GuidedHamiltonianPhaseEstimation
    )

    guided = source._guided_hamiltonian_symb.make()
    original = guided.qpe_bloq
    from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState

    assert isinstance(original.qpe_window_state, KaiserWindowState)
    modeled = GuidedHamiltonianModel(
        *_widths(guided),
        guided.hamiltonian.alpha,
        guided.lambd,
        guided.alpha,
        guided.gamma,
    )
    assert symbolic_guided_hamiltonian() == modeled
    assert modeled.phase_estimation.precision == guided.qpe_precision
    assert modeled.phase_estimation.failure_probability == guided.qpe_fail_prob
    assert modeled.amplification_rounds == guided.n_rounds_amplification
    assert modeled.phase_estimation.window_alpha == original.qpe_window_state.alpha
    _assert_signature(modeled.signature, guided)

    qpe = original.qpe_bloq
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    reflection = qpe.walk.reflect.controlled(ctrl_spec=CtrlSpec(cvs=0))
    assert (
        isinstance(reflection, ReflectionUsingPrepare) and reflection.control_val == 0
    )
    assert isinstance(qpe.qft_inv, Adjoint)
    assert (
        isinstance(qpe.qft_inv.subbloq, QFTTextBook)
        and qpe.qft_inv.subbloq.with_reverse
    )
    assert (
        guided._refl_guide_ancilla.global_phase == guided._refl_all.global_phase == -1
    )
    assert [r.bitsize for r in guided._refl_guide_ancilla.signature] == [
        guided.guiding_state.junk_bitsize
    ]
    assert guided._refl_all.signature == guided.signature
    mapping = {
        guided: _ROOT,
        original: _PHASE,
        original.adjoint(): f"{_PHASE}†",
        guided._refl_guide_ancilla: "ReflectGuideAncilla(-1)",
        guided._refl_all: "ReflectAll(-1)",
        original.guiding_state: "BlackBoxPrepare",
        original.guiding_state.adjoint(): "BlackBoxPrepare†",
        qpe: _QPE,
        qpe.adjoint(): f"{_QPE}†",
        **_qpe_keys(qpe),
        **_qpe_keys(qpe, adjoint=True),
    }
    # Forward and adjoint reflection identify the same source operation.
    roots = (guided, original, original.adjoint(), qpe, qpe.adjoint())
    assert modeled.source_call_graph().direct_calls == {
        (mapping[parent], mapping[child]): count
        for parent in roots
        for child, count in get_bloq_callee_counts(parent, ignore_decomp_failure=False)
    }
    for parent in (guided, original):
        with pytest.raises(DecomposeNotImplementedError):
            parent.decompose_bloq()


@pytest.mark.parametrize(
    "precision,probability", [(None, None), (0.125, 0.01), (0.0625, 0.125)]
)
def test_independent_phase_root(
    precision: float | None, probability: float | None
) -> None:
    from attrs import evolve
    from qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian import (
        _guided_phase_estimate_symb,
    )
    from qualtran.resource_counting._call_graph import get_bloq_callee_counts

    original = _guided_phase_estimate_symb.make()
    if precision is not None:
        original = evolve(original, precision=precision, fail_prob=probability)
    model = GuidedPhaseEstimationModel(
        *_widths(original), original.precision, original.fail_prob
    )
    if precision is None:
        assert symbolic_guided_phase_estimation() == model
    from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState

    assert isinstance(original.qpe_window_state, KaiserWindowState)
    assert model.window_alpha == original.qpe_window_state.alpha
    assert model.phase_width == original.qpe_window_state.bitsize
    _assert_signature(model.signature, original)
    qpe = original.qpe_bloq
    keys = {original.guiding_state: "BlackBoxPrepare", qpe: _QPE, **_qpe_keys(qpe)}
    assert model.source_call_graph().direct_calls == {
        (parent, keys[child]): count
        for parent, declaration in ((_PHASE, original), (_QPE, qpe))
        for child, count in get_bloq_callee_counts(
            declaration, ignore_decomp_failure=False
        )
    }
    assert model.source_call_graph(max_depth=1).leaf_counts == {
        "BlackBoxPrepare": 1,
        _QPE: 1,
    }


@pytest.mark.parametrize(
    "resource,selection,message", [(1, 2, "resource"), (0, 3, "match")]
)
def test_guided_model_rejects_unsupported_widths(
    resource: int, selection: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GuidedPhaseEstimationModel(2, 1, selection, 0, 0.1, 0.01, resource)
    with pytest.raises(ValueError, match=message):
        GuidedHamiltonianModel(2, 1, selection, 0, 1, 0.25, 0.5, 0.5, resource)


def test_symbolic_factories_use_actual_native_query_owners() -> None:
    from qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian import (
        _guided_hamiltonian_symb,
    )
    from sympy import symbols

    from examples.algorithms.guided_hamiltonian_models import _problem
    from flagquantum.algorithms.kikuchi_guiding_state import GuidingState

    n, k, m, c = symbols("n k m c", positive=True, integer=True)
    zeta = symbols(r"\zeta", positive=True)
    block, guide = _problem(n, k, c, (1 - zeta) * m, zeta * m)
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_block_encoding import (
        KikuchiHamiltonian as SourceHamiltonian,
    )
    from qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state import (
        GuidingState as SourceGuide,
    )

    source = _guided_hamiltonian_symb.make()
    assert isinstance(source.hamiltonian, SourceHamiltonian)
    assert isinstance(source.guiding_state.prepare, SourceGuide)
    assert isinstance(guide.prepare, GuidingState)
    assert guide.prepare.inst.m == source.guiding_state.prepare.inst.m
    assert block.inst.m == source.hamiltonian.inst.m
    # The original third positional argument is entry precision, not sparsity.
    assert block.entry_bitsize == source.hamiltonian.entry_bitsize
    assert block.s == source.hamiltonian.s == block.inst.num_unique_constraints
    assert block.alpha == source.hamiltonian.alpha
    assert block.epsilon == source.hamiltonian.epsilon
    assert block.resource_bitsize == source.hamiltonian.resource_bitsize == 0
    assert guide.selection_width == source.guiding_state.selection_bitsize
    assert guide.junk_width == source.guiding_state.junk_bitsize
