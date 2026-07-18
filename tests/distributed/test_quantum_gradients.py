# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

if os.environ.get("HAS_MTHREADS") == "1":
    try:
        import torchada  # noqa: F401

        print("✓ torchada imported for Mthreads GPU")
    except ImportError:
        print("⚠ torchada not available, falling back to default")

import torch
from torch.distributed.tensor import DTensor, Partial

import flagquantum as fq


def test_inv(verbose=False):
    """
    invertible gradient
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2

    qdev = fq.DistributedQuantumDevice(
        nq,
        device="cuda",
        world_sz=world_sz,
        invertible=False,
    )
    qdev_i = fq.DistributedQuantumDevice(
        nq,
        device="cuda",
        world_sz=world_sz,
        invertible=True,
    )

    # test registration
    fq.ops.register_gate("id", torch.eye(2, dtype=torch.cfloat))

    def fun(qdev, inp):
        func_list = [
            {"func": "ry", "wires": [0], "input_idx": [0]},
            {"func": "ry", "wires": [1], "input_idx": [1]},
            {"func": "ry", "wires": [2], "input_idx": [2]},
        ]
        enc = fq.GeneralEncoder(func_list)
        base_mod = (
            [enc]
            + [fq.CX(wires=[i, (i + 1) % nq]) for i in range(qdev.n_wires)]
            + [fq.ops.registry.ID(wires=[1])]
        )
        if qdev.invertible:
            mod = fq.invertible.InvertibleUnitary(base_mod)
            mod.train()
            mod(qdev, inp)
        else:
            enc(qdev, inp)
            if inp.device.index == 0:
                print(f"  after enc: {qdev._states}; wire order: {qdev._wire_order}")
            for m_ in base_mod[1:]:
                m_.train()
                m_(qdev)
                if inp.device.index == 0:
                    print(
                        f"  after {type(m_).__name__}: {qdev._states}; wire order: {qdev._wire_order}"
                    )
        # return qdev._states
        meas_approx = fq.measure_allZ(qdev, shots=0, training=True)
        return meas_approx

    if verbose:
        print(f"after {rank} {qdev.states}")
        print(f"done {qdev.states.full_tensor()}")

    # test backprop:
    x_i = torch.nn.Parameter(
        torch.tensor([[torch.pi / 3, -torch.pi / 3, torch.pi / 6]])
    )

    out_i = fun(qdev_i, x_i)
    print(f"states invertible: {torch.view_as_complex(qdev_i.states.full_tensor())}")
    print(f"out invertible: {out_i}")
    out_i.abs().sum().backward()
    x_i_grad_dist = DTensor.from_local(
        x_i.grad, qdev_i.device_mesh, placements=[Partial()]
    )
    x_i_grad = x_i_grad_dist.full_tensor()

    # this is for when out is qdev.state (not as good a test, since it doesn't check ordering):
    # assert torch.allclose(x_grad.cpu(), torch.tensor([[ 0.30618620, -0.30618620,  0.65973961]]))
    # this is for when out is noiseless measurement:
    print(x_i_grad)

    x = torch.nn.Parameter(torch.tensor([[torch.pi / 3, -torch.pi / 3, torch.pi / 6]]))
    out = fun(qdev, x)
    print(f"states: {torch.view_as_complex(qdev.states.full_tensor())}")
    print(f"out: {out}")
    out.abs().sum().backward()
    x_grad_dist = DTensor.from_local(x.grad, qdev.device_mesh, placements=[Partial()])
    x_grad = x_grad_dist.full_tensor()
    print(x_grad)

    assert torch.allclose(
        torch.view_as_complex(qdev.states.full_tensor().cpu()),
        torch.tensor(
            [
                [
                    [[0.7244 + 0.0j, -0.0647 + 0.0j], [-0.1121 + 0.0j, 0.4183 + 0.0j]],
                    [[-0.2415 + 0.0j, 0.1941 + 0.0j], [0.1121 + 0.0j, -0.4183 + 0.0j]],
                ]
            ]
        ),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(
        x_grad.cpu(), torch.tensor([[-0.80801272, 1.55801260, -0.37500000]])
    )
    if rank == "0":
        print("inverse test passed!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_inv(False)
