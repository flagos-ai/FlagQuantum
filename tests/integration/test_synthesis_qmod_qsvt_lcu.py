"""Check phase conventions, coherent parity selection, and borrowed QSVT ports."""

import cmath
import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
_DTYPE = torch.complex128
_SOURCE = """
qfunc signal(a: qbit) {
  RY(2 ** (-2), a);
}
qfunc main(output sig: qbit, output aux: qbit, output parity: qbit) {
  allocate(sig); allocate(aux); allocate(parity);
  H(sig); H(parity); RZ(0.37, parity);
  qsvt_lcu(EVEN, ODD,
    lambda(a) { a ^= sig == 0; },
    lambda(a) { a ^= sig == 1; },
    lambda() { signal(sig); },
    aux, parity);
}
"""


def _reference(even: list[float], odd: list[float]) -> torch.Tensor:
    circuit = fq.Circuit(3, dtype=_DTYPE).h(0).h(2).rz(2, theta=0.37).h(1)
    c, s = math.cos(0.125), math.sin(0.125)
    block = torch.tensor([[c, -s], [s, c]], dtype=_DTYPE)
    identity = torch.eye(2, dtype=_DTYPE)
    for index in range(max(len(even), len(odd))):
        if index:
            rotation = block if index % 2 else block.mH
            circuit.any(
                2,
                0,
                matrix=torch.block_diag(
                    rotation if index < len(even) else identity,
                    rotation if index < len(odd) else identity,
                ),
            )
        diagonal = []
        for sig in (0, 1):
            for aux in (0, 1):
                projector = sig == int(index % 2 == 1)
                sign = 1 if aux ^ projector else -1
                for branch in (even, odd):
                    angle = branch[index] if index < len(branch) else 0
                    diagonal.append(cmath.exp(0.5j * sign * angle))
        circuit.any(0, 1, 2, matrix=torch.diag(torch.tensor(diagonal, dtype=_DTYPE)))
    return fq.run(circuit.h(1)).statevector()


@pytest.mark.parametrize(
    "even,odd", [([0.23], [-0.4, 0.8]), ([0.53, -0.12, 1.1], [0.7, -0.9])]
)
def test_qsvt_lcu_keeps_both_parities_and_absolute_phase(
    even: list[float], odd: list[float]
) -> None:
    source = _SOURCE.replace("EVEN", repr(even)).replace("ODD", repr(odd))
    program = synth.synthesize(
        source, source_format="qmod", max_qubits=3, max_workspace=0
    )
    assert dict(program.registers) == {"sig": (0,), "aux": (1,), "parity": (2,)}
    torch.testing.assert_close(
        fq.run(program).statevector(), _reference(even, odd), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "old,new,diagnostic",
    [
        ("EVEN", "[0.23, 0.44]", "adjacent even and odd"),
        ("aux, parity", "aux, aux", "must be disjoint"),
        ("sig == 0", "parity == 0", "cannot read the selector"),
        ("signal(sig)", "signal(aux)", "block cannot use"),
        ("signal(sig)", "allocate(sig)", "requires pure effects"),
        ("2 ** (-2)", "0 ** (-2)", "negative real powers require"),
        ("2 ** (-2)", "2 ** (-256)", "negative real powers require"),
    ],
)
def test_qsvt_lcu_rejects_invalid_callbacks_and_bounds(
    old: str, new: str, diagnostic: str
) -> None:
    source = (
        _SOURCE.replace(old, new).replace("EVEN", "[0.2]").replace("ODD", "[0.3, 0.4]")
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
