import os

import pytest

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_accel,
    pytest.mark.distributed_multinode,
    pytest.mark.gpu,
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="requires torchrun with distributed RANK/WORLD_SIZE environment",
    ),
]

if os.environ.get("HAS_MTHREADS") == "1":
    try:
        import torchada  # noqa: F401

        print("✓ torchada imported for Mthreads GPU")
    except ImportError:
        print("⚠ torchada not available, falling back to default")

import torch
from torch.distributed.tensor import DTensor, Partial

import flagquantum as fq


def test_groupings(verbose=False):
    """
    Testing qubit groupings inside dtensor
    """
    rank = os.environ["RANK"]
    max_dtensor_dims = 7
    nq = 7
    wire = 1
    world_sz = 2

    qdev = fq.DistributedQuantumDevice(
        nq, device="cuda", world_sz=world_sz, max_dtensor_dims=max_dtensor_dims
    )

    if verbose:
        print(f"before {rank} {qdev.states}")

    # test class method
    qdev.y(wires=[wire])
    if verbose:
        print(f"after y {rank} {qdev.states}")

    # test Module
    cx_gate = fq.CX(wires=[wire, (wire + 3) % nq])
    cx_gate(qdev)
    if verbose:
        print(f"after cx {rank} {qdev.states}")

    # test functional
    fq.rz(qdev, wires=[(wire + 2) % nq], params=torch.pi / 3)
    if verbose:
        print(f"after rz {rank} {qdev.states}")

    # test registration
    fq.ops.register_gate("id", torch.eye(2, dtype=torch.cfloat))
    fq.ops.registry.id(qdev, wires=[1])

    if verbose:
        print(f"after {rank} {qdev.states}")
        print(f"done {qdev.states.full_tensor()}")

    qdev.canonicalize()
    states_tq = torch.Tensor(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [
                                        [0.5000, torch.sqrt(torch.Tensor([3])) / 2],
                                        [0.0000, 0.0000],
                                    ],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                            [
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                                [
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                                ],
                            ],
                        ],
                    ],
                ],
            ]
        ]
    )

    assert torch.allclose(
        states_tq, qdev.states.full_tensor().cpu()[0].reshape((1,) + (2,) * nq + (2,))
    )

    # test noisy non-differentiable (sampling) measurement
    meas_samp = fq.measure_allZ(qdev, shots=int(1e5), training=False)  # noqa: F841

    # test noisy differentiable (approximate) measurement
    meas_approx = fq.measure_allZ(qdev, shots=int(1e5), training=True)  # noqa: F841

    if rank == "0":
        print("groupings test passed!")


def test_dqd(verbose=False):
    """
    monotest
    """
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2
    wire = 1

    qdev = fq.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    if verbose:
        print(f"before {rank} {qdev.states}")

    # test class method
    qdev.y(wires=[wire])
    if verbose:
        print(f"after y {rank} {qdev.states}")

    # test Module
    cx_gate = fq.CX(wires=[wire, (wire + 1) % nq])
    cx_gate(qdev)
    if verbose:
        print(f"after cx {rank} {qdev.states}")

    # test functional
    fq.rz(qdev, wires=[wire], params=torch.pi / 3)
    if verbose:
        print(f"after rz {rank} {qdev.states}")

    # test registration
    fq.ops.register_gate("id", torch.eye(2, dtype=torch.cfloat))
    fq.ops.registry.id(qdev, wires=[1])

    if verbose:
        print(f"after {rank} {qdev.states}")
        print(f"done {qdev.states.full_tensor()}")

    """
    # compare against ground truth
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.y(wires=[wire])
    qdev_tq.cx(wires=[wire, (wire+1) % nq])
    qdev_tq.rz(wires=[wire], params=torch.pi/3)

    # remove singleton batch dimension and put complex split dimension in front to match our implementation
    states_tq = torch.view_as_real(qdev_tq.states).permute([0,4,1,2,3])[0]
    """
    states_tq = torch.Tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [-0.5, torch.sqrt(torch.tensor([3])) / 2]],
            ],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]
    )

    if verbose:
        print(f"torchquantum {states_tq}")
    assert torch.allclose(states_tq, qdev.states.full_tensor().cpu()[0])
    if rank == "0":
        print(
            "class method, module, functional, registration, and correctness test passed!"
        )

    # test noiseless measurement
    meas = fq.measure_allZ(qdev)
    """
    meas_tq = tq.MeasureAll(tq.PauliZ)(qdev_tq)
    """
    meas_tq = torch.Tensor([[1.0, -1.0, -1.0]])
    if verbose:
        print(f"meas {rank} {meas} {meas_tq}")
    assert torch.allclose(meas_tq, meas.cpu())
    if rank == "0":
        print("noiseless measurement passed!")

    # test noisy non-differentiable (sampling) measurement
    meas_samp = fq.measure_allZ(qdev, shots=1024, training=False)
    if verbose:
        print(f"meas_noisy non-diff {rank} {meas_samp}")
    assert torch.allclose(meas_tq, meas_samp.cpu())

    # test noisy differentiable (approximate) measurement
    meas_approx = fq.measure_allZ(qdev, shots=1024, training=True)
    if verbose:
        print(f"meas_noisy diff {rank} {meas_approx}")
    assert torch.allclose(meas_tq, meas_approx.cpu())

    if rank == "0":
        print("monotest noisy measurement test passed!")


def test_noisy_meas(verbose=False):
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2
    wire = 1

    qdev = fq.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    qdev.rx(wires=[wire], params=torch.pi / 6)

    """
    qdev_tq = tq.QuantumDevice(3)
    qdev_tq.rx(wires=[wire], params=torch.pi/6)
    meas_tq = tq.MeasureAll(tq.PauliZ)(qdev_tq)
    """
    meas_tq = torch.Tensor([[1.0, torch.sqrt(torch.tensor([3])) / 2, 1.0]])

    # test noisy non-differentiable (sampling) measurement
    meas_samp, _ = fq.measure_allZ(
        qdev, shots=int(1e5), training=False, postselect_cond={2: 0}
    )

    # test noisy differentiable (approximate) measurement
    meas_approx = fq.measure_allZ(qdev, shots=int(1e5), training=True)
    if verbose:
        print(f"meas_noisy {rank} {meas_tq} {meas_samp} {meas_approx}")

    assert torch.allclose(meas_tq, meas_approx.cpu(), rtol=1e-3, atol=1e-2)
    assert torch.allclose(meas_tq, meas_samp.cpu(), rtol=1e-3, atol=1e-2)

    if rank == "0":
        print("standalone noisy measurement test passed!")


def test_encoder(verbose=False):
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2

    qdev = fq.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)

    func_list = [
        {"func": "rx", "wires": [0], "input_idx": [0]},
        {"func": "ry", "wires": [1], "input_idx": [1]},
        {"func": "rz", "wires": [2], "input_idx": [2]},
    ]
    x = torch.Tensor([[1, 2, 3]])
    # test encoder
    enc = fq.GeneralEncoder(func_list)
    enc(qdev, x)
    states = qdev.states

    # compare against ground truth
    """
    import torchquantum as tq
    qdev_tq = tq.QuantumDevice(3)
    enc_tq = tq.GeneralEncoder(func_list)
    enc_tq(qdev_tq, x)
    states_tq = torch.view_as_real(qdev_tq.states)
    """
    states_tq = torch.Tensor(
        [
            [
                [
                    [[0.03354074, -0.47297207], [0.00000000, 0.00000000]],
                    [[0.05223661, -0.73661041], [0.00000000, 0.00000000]],
                ],
                [
                    [[-0.25838584, -0.01832339], [0.00000000, 0.00000000]],
                    [[-0.40241212, -0.02853699], [0.00000000, 0.00000000]],
                ],
            ]
        ]
    )
    if verbose:
        print(f"  enc torchquantum {states_tq}\n  enc tqd {states.full_tensor()}")

    assert torch.allclose(states_tq, states.full_tensor().cpu())

    if rank == "0":
        print("standalone encoder test passed!")


def test_grads(verbose=False):
    rank = os.environ["RANK"]
    nq = 3
    world_sz = 2

    qdev = fq.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
    p = torch.nn.Parameter(torch.Tensor([torch.pi / 3, -torch.pi / 3, torch.pi / 6]))
    [qdev.ry(wires=[i], params=p[i]) for i in range(nq)]
    if verbose:
        print(f"after ry {rank} {qdev.states}")
    [qdev.cx(wires=[i, (i + 1) % nq]) for i in range(nq)]

    qdev.states.abs().sum().backward()
    grad_dist = DTensor.from_local(
        p.grad[None], qdev.device_mesh, placements=[Partial()]
    )
    grad = grad_dist.full_tensor()
    assert torch.allclose(
        grad.cpu(),
        torch.Tensor([0.3061861991882324, -0.3061861991882324, 0.65973961353302]),
    )
    if rank == "0":
        print("grad test passed!")


if __name__ == "__main__":
    test_dqd(False)
    test_noisy_meas(False)
    test_encoder(False)
    test_grads(False)
    test_groupings(False)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
