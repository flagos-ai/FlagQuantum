"""Original occurrence selection and same-path contraction-info accounting."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.algorithms.conjunction_contraction import (
    contract_conjunction_with_info,
    main,
    select_conjunction_instances,
)
from examples.algorithms.conjunction_tensors import conjunction_providers
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.contraction import (
    _pair_steps_from_dynamic_path,
    _profile_from_steps,
)
from flagquantum.simulation.tensor_network.register_axis_groups import (
    register_tensor_label_groups,
)
from flagquantum.simulation.tensor_network.stages import execute_pair_steps

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = [
    (1, 1, 1),
    (1, 0, 1),
    (1, 1, 1, 1),
    (1, 0, 1, 0),
    (0, 0, 0, 0),
    (1, 1, 1, 1, 1),
]


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("keep", [1, 2, 3, 999])
def test_original_occurrence_ids_leaf_axes_dense_and_same_path_info(
    values: tuple[int, ...], keep: int
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.simulation.tensor import cbloq_to_quimb

    source = MultiAnd(values)
    original = source.as_composite_bloq().flatten(lambda instance: instance.i != keep)
    source_instances = tuple(instance.i for instance, _, _ in original.iter_bloqsoqs())
    like = torch.empty((), dtype=torch.complex128)
    selected, labels = select_conjunction_instances(values, like, keep_instance=keep)
    assert labels == source_instances == tuple(range(1, len(values)))
    assert tuple(
        op.callee for op in selected.functions[0].blocks[0].operations
    ) == tuple(
        f"and_{a.cv1}{a.cv2}" for i, _, _ in original.iter_bloqsoqs() for a in (i.bloq,)
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    providers = conjunction_providers(values, like)
    nodes, inc, out = component_tensor_nodes(
        selected, order, providers.supply, like, "tensor_root"
    )
    tn = cbloq_to_quimb(original)
    for native, reference in zip(nodes, tn.tensors, strict=True):
        np.testing.assert_array_equal(native.tensor.numpy(), reference.data)
    signature = Signature(selected.functions[0].registers or ())
    groups = register_tensor_label_groups(signature, inc, out)
    outputs = tuple(label for group in groups for label in group)
    info = tn.contraction_info(optimize="greedy")
    # Reuse the existing imported-path adapter and executor. This holds path
    # choice fixed when comparing the two libraries' distinct cost conventions.
    steps = _pair_steps_from_dynamic_path(nodes, outputs, info.path)
    profile = _profile_from_steps(nodes, outputs, strategy="source-greedy", steps=steps)
    assert profile.peak_size == int(info.largest_intermediate)
    assert profile.n_steps == len(info.path) == len(values) - 2
    assert profile.output_size == 2 ** (3 * len(values) - 1)
    # These networks use pairwise binary sums: opt_einsum counts multiply+add
    # as two FLOPs; native estimates one union-dimension product per entry.
    assert int(info.opt_cost) == 2 * profile.estimated_cost
    contracted = execute_pair_steps(nodes, outputs, steps).reshape(
        2 ** (2 * len(values) - 1), 2 ** len(values)
    )
    np.testing.assert_array_equal(contracted.numpy(), source.tensor_contract())
    matrix, greedy = contract_conjunction_with_info(values, like, keep_instance=keep)
    torch.testing.assert_close(matrix, contracted, rtol=0, atol=0)
    assert greedy.output_size == matrix.numel()
    assert greedy.n_steps == len(nodes) - 1
    # A tensor-level phase keeps the gradient graph through imported-path
    # execution. Its derivative is compared to the exact already checked map.
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    phased = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    result = execute_pair_steps(phased, outputs, steps).reshape(matrix.shape)
    expected = matrix * torch.exp(1j * phase)
    torch.testing.assert_close(result, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(result.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
        rtol=0,
        atol=2e-12,
    )


def test_root_retention_fails_without_logical_multiand_provider() -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.simulation.tensor import cbloq_to_quimb

    original = MultiAnd((1,) * 4).as_composite_bloq().flatten(lambda i: i.i != 0)
    with pytest.raises(NotImplementedError):
        cbloq_to_quimb(original)
    with pytest.raises(ValueError, match="no tensor provider.*multi_and"):
        contract_conjunction_with_info(
            (1,) * 4, torch.empty((), dtype=torch.complex128), keep_instance=0
        )


def test_shape_only_profile_on_meta_matches_real_network() -> None:
    values = (1, 0, 1, 0)
    cpu, expected = contract_conjunction_with_info(
        values, torch.empty((), dtype=torch.complex128)
    )
    meta, actual = contract_conjunction_with_info(
        values, torch.empty((), dtype=torch.complex128, device="meta")
    )
    assert meta.device.type == "meta" and meta.shape == cpu.shape
    assert actual == expected
    record, _ = select_conjunction_instances(values, torch.empty(()))
    assert len(record.functions[0].blocks[0].operations) == 3


def test_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/conjunction_contraction.py"
        ),
        run_name="__main__",
    )
    text = capsys.readouterr().out
    assert "instances: (1, 2, 3)" in text
    assert "peak=2048 elements, steps=2" in text
