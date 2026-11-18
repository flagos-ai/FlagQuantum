"""Whole Phase source binding, including zero logical ports and scalar carriers."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_phase_source() -> None:
    import kickmix
    from qualtran.bloqs.block_encoding import phase

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, blob in (
        ("phase.py", "d7e6b03237f63238e511a4e7aa5056808a646593"),
        ("phase_test.py", "c41738844b4eb08517b4c3245d7cc47f1d6bdf1b"),
        ("phase.ipynb", "c832c8d59cbbfcebd8210010b9aa44af23655ec4"),
    ):
        data = (Path(phase.__file__).parent / name).read_bytes()
        assert sha1(b"blob %d\0" % len(data) + data).hexdigest() == blob


def scalar_source(
    exponent: float = -0.2, auxiliary: int = 0, empty_identity: bool = True
) -> Any:
    from attrs import frozen
    from qualtran import QAny, Signature
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    @frozen
    class Scalar(BlockEncoding):
        @property
        def signature(self) -> Any:
            return Signature.build_from_dtypes(ancilla=QAny(auxiliary))

        @property
        def system_bitsize(self) -> int:
            return 0

        @property
        def ancilla_bitsize(self) -> int:
            return auxiliary

        @property
        def resource_bitsize(self) -> int:
            return 0

        @property
        def alpha(self) -> float:
            return 1.0

        @property
        def epsilon(self) -> float:
            return 0.0

        @property
        def signal_state(self) -> Any:
            return BlackBoxPrepare(
                PrepareIdentity.from_bitsizes(
                    [] if empty_identity and not auxiliary else [auxiliary]
                )
            )

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            bb.add(GlobalPhase(exponent=exponent))
            return soqs

        def build_call_graph(self, ssa: Any) -> dict[Any, int]:
            return {GlobalPhase(exponent=exponent): 1}

    return Scalar()


@pytest.mark.parametrize("empty_identity", (False, True))
@pytest.mark.parametrize("auxiliary", (0, 1))
@pytest.mark.parametrize("controlled,adjoint", tuple(product((False, True), repeat=2)))
def test_actual_scalar_blocks_use_owned_snapshots_with_all_four_interfaces(
    empty_identity: bool, auxiliary: int, controlled: bool, adjoint: bool
) -> None:
    source = scalar_source(auxiliary=auxiliary, empty_identity=empty_identity)
    block = import_block_encoding(source, models=original_models(source))
    assert block.system_bitsize == 0
    assert block.alpha == source.alpha and block.epsilon == source.epsilon
    offset = int(controlled)
    logical = offset + auxiliary
    work, slots = block.requirements(controlled=controlled)
    assert (work, slots) == (int(not logical), 0)
    size = logical + work
    operation = block.make_on(
        system=(),
        ancilla=tuple(range(offset, logical)),
        control=(0,) if controlled else (),
        workspace=tuple(range(logical, size)),
        adjoint=adjoint,
    )
    ir = Circuit(size).append(operation).to_ir()
    original = source.controlled() if controlled else source
    original = original.adjoint() if adjoint else original
    expected = torch.as_tensor(
        original.tensor_contract(), dtype=torch.complex128
    ).reshape(1 << logical, 1 << logical)
    if work:
        expected = expected[0, 0] * torch.eye(2, dtype=torch.complex128)
    restored = CircuitIR.from_json(ir.to_json())
    assert restored.to_json() == ir.to_json()
    torch.testing.assert_close(
        get_unitary(restored, dtype=torch.complex128), expected, atol=4e-12, rtol=0
    )
    assert_source_model(
        block.resource_model(controlled=controlled, adjoint=adjoint), original
    )
    assert len(ir.instructions) == (1 if controlled else 4)
    assert not any(g.name in {"measure", "reset"} for g in ir.instructions)
    # A carrier in arbitrary state is preserved up to the same scalar phase.
    if work:
        assert all(g.wires == (0,) for g in ir.instructions)


@pytest.mark.parametrize("phi", (0.0, 0.25, -1 / 3))
@pytest.mark.parametrize("exponent", (0.0, -0.2))
@pytest.mark.parametrize("adjoint", (False, True))
def test_existing_phase_consumer_replaces_zero_port_child_and_preserves_carrier(
    phi: float, exponent: float, adjoint: bool
) -> None:
    from qualtran.bloqs.block_encoding.phase import Phase

    source_child = scalar_source(exponent)
    child = import_block_encoding(source_child, models=original_models(source_child))
    work, slots = child.requirements()
    assert not slots
    native = PhaseBlockEncoding(child, phi, 0)
    bound_child = child.make_on(
        system=(), workspace=tuple(range(work)), adjoint=adjoint
    )
    ir = (
        Circuit(max(1, work))
        .append(native.make_on(bound_child, anchor=0, adjoint=adjoint))
        .to_ir()
    )
    source = Phase(source_child, phi, 0)
    original = source.adjoint() if adjoint else source
    expected = complex(source.tensor_contract())
    if adjoint:
        expected = expected.conjugate()
    matrix = get_unitary(CircuitIR.from_json(ir.to_json()), dtype=torch.complex128)
    torch.testing.assert_close(
        matrix, expected * torch.eye(2, dtype=torch.complex128), atol=4e-12, rtol=0
    )
    assert tuple(native.signature) == ()
    assert len(ir.instructions) == 4 * (int(bool(exponent)) + int(bool(phi)))
    assert_source_model(
        native.resource_model(child.resource_model(adjoint=adjoint)), original
    )
    assert (
        estimate_component_resources(
            native.resource_model(child.resource_model())
        ).peak_logical_qubits
        == 0
    )
    if exponent:
        # Source calls and output order are captured as owned IR independently.
        whole = import_block_encoding(source, models=original_models(source))
        total_work, _ = whole.requirements()
        root = (
            Circuit(total_work)
            .append(
                whole.make_on(
                    system=(), workspace=tuple(range(total_work)), adjoint=adjoint
                )
            )
            .to_ir()
        )
        torch.testing.assert_close(
            get_unitary(root, dtype=torch.complex128), matrix, atol=4e-12, rtol=0
        )


def test_empty_identity_scalar_and_explicit_capture_admission() -> None:
    from attrs import frozen
    from qualtran import Signature
    from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    @frozen
    class Identity(BlockEncoding):
        signature = Signature(())
        alpha = 1.0
        epsilon = 0.0
        system_bitsize = ancilla_bitsize = resource_bitsize = 0

        @property
        def signal_state(self) -> Any:
            return BlackBoxPrepare(PrepareIdentity.from_bitsizes([]))

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            return soqs

        def build_call_graph(self, ssa: Any) -> dict[Any, int]:
            return {}

    child = import_block_encoding(Identity(), include_controlled=False)
    assert child.requirements() == (1, 0)
    op = child.make_on(system=(), workspace=(0,))
    assert op.required_qubits == 1
    assert Circuit(1).append(op).to_ir().instructions == ()
    with pytest.raises(ValueError, match="declared anchor"):
        _snapshot(scalar_source(), ())
    with pytest.raises(ValueError, match="empty logical ports"):
        _snapshot(scalar_source(), (("fake", 1),), scalar_carrier=True)
    with pytest.raises(ValueError, match="empty logical ports"):
        _snapshot(scalar_source(), (), scalar_carrier=1)
    # Neither the plain snapshot nor its costs imply an unrequested control.
    with pytest.raises(ValueError, match="not captured"):
        child.requirements(controlled=True)


def test_original_factory_replaces_phase_consumer_and_all_source_definitions() -> None:
    from qualtran.bloqs.block_encoding.phase import _PHASE_DOC, _phase_block_encoding
    from sympy import Symbol

    source = _phase_block_encoding()
    child = import_block_encoding(
        source.block_encoding, models=original_models(source.block_encoding)
    )
    native = PhaseBlockEncoding(child, source.phi, source.eps)
    assert tuple(_PHASE_DOC.examples) == (_phase_block_encoding,)
    assert native.label(str(source.block_encoding)[2:-1]) == str(source)
    symbolic = PhaseBlockEncoding(child, Symbol("phi", real=True), 0)
    symbolic_source = type(source)(source.block_encoding, symbolic.phi, 0)
    assert symbolic.label(str(source.block_encoding)[2:-1]) == str(symbolic_source)
    for adjoint in (False, True):
        original = source.adjoint() if adjoint else source
        ir = (
            Circuit(1)
            .append(
                native.make_on(
                    child.make_on(system=(0,), adjoint=adjoint),
                    anchor=0,
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128),
            torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
            atol=4e-12,
            rtol=0,
        )
        assert len(ir.instructions) == 5
        assert_source_model(
            native.resource_model(child.resource_model(adjoint=adjoint)), original
        )
    whole = import_block_encoding(source, models=original_models(source))
    for controlled, adjoint in product((False, True), repeat=2):
        original = source.controlled() if controlled else source
        original = original.adjoint() if adjoint else original
        ir = (
            Circuit(2 if controlled else 1)
            .append(
                whole.make_on(
                    system=(1,) if controlled else (0,),
                    control=(0,) if controlled else (),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128),
            torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
            atol=4e-12,
            rtol=0,
        )
        assert_source_model(
            whole.resource_model(controlled=controlled, adjoint=adjoint), original
        )
