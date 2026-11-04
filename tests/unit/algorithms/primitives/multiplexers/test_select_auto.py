"""Check automatic SELECT lowering through the user execution API."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU, SelectSparse

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


def test_sparse_select_uses_dynamic_sampling_with_shared_workspace() -> None:
    circuit = fq.Circuit(11)
    circuit.append(
        SelectSparse.make_on(
            p=(0,),
            q=(1,),
            r=(2,),
            s=(3,),
            alpha=4,
            beta=5,
            flag_1b=6,
            sys=(7, 8, 9, 10),
        )
    )
    sampled = fq.run(
        circuit, options=fq.ExecutionOptions(target="samples", shots=4, seed=5)
    )
    assert sampled.plan is not None
    report = sampled.plan.summary()["select_lowering"]
    assert report["workspace_qubits"] == 3
    assert [operation["path"] for operation in report["operations"]] == [
        "dynamic_unary_iteration"
    ] * 4
    torch.testing.assert_close(
        sampled.require_samples(), torch.zeros(4, 11, dtype=torch.int64)
    )
