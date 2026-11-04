"""Check automatic SELECT lowering through the user execution API."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU

pytestmark = pytest.mark.unit


def test_run_reports_dense_and_dynamic_select() -> None:
    circuit = fq.Circuit(3)
    # Address two selects X on the data qubit. Remap the operands to
    # exercise the compiler's mapping and leave a visible final result.
    circuit.x(2)
    circuit.append(
        SelectPauliLCU(2, 1, ("I", "Z", "X", "-Y")).make_on(
            selection=(2, 0), target=(1,)
        )
    )
    dense = fq.run(circuit)
    assert dense.plan is not None
    assert dense.plan.summary()["select_lowering"]["operations"][0]["path"] == "dense"

    sampled = fq.run(
        circuit, options=fq.ExecutionOptions(target="samples", shots=12, seed=5)
    )
    assert sampled.plan is not None
    report = sampled.plan.summary()["select_lowering"]
    assert report["operations"][0]["path"] == "dynamic_unary_iteration"
    assert report["workspace_qubits"] == 1
    expected = torch.tensor([[0, 1, 1]]).expand(12, -1)
    torch.testing.assert_close(sampled.require_samples(), expected)
    assert len(sampled.measurements) == 1

    # A serialized plan carries the chosen implementation and logical width.
    restored = type(sampled.plan).from_dict(sampled.plan.to_dict())
    torch.testing.assert_close(fq.run(restored).require_samples(), expected)
