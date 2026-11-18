"""Exact external source recipes needed by complete Chebyshev block binding."""

from collections import Counter
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
