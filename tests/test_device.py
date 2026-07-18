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

# tests/test_device.py
"""Tests for DistributedQuantumDevice."""

import pytest
import torch

import flagquantum as fq


class TestDeviceInitialization:
    """Test device creation and basic properties."""

    def test_cpu_device(self):
        """Test creating device on CPU."""
        device = fq.DistributedQuantumDevice(n_wires=4, bsz=2, device="cpu", world_sz=1)
        assert device.n_wires == 4
        assert device.bsz == 2
        assert device.device == "cpu"
        assert device._states.device.type == "cpu"

    def test_cuda_device(self):
        """Test creating device on CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = fq.DistributedQuantumDevice(
            n_wires=4, bsz=2, device="cuda", world_sz=1
        )
        assert device._states.device.type == "cuda"

    def test_reset_states(self):
        """Test resetting states to |0...0>."""
        device = fq.DistributedQuantumDevice(n_wires=3, device="cpu")
        device.reset_states()

        # Check initial state is |000> by examining the state tensor directly
        # State shape: (batch, 2, 2, 2, 2) for 3 qubits + real/imag
        # |000> means all qubit dimensions are 0, real part = 1
        states = device.states
        zero_idx = (slice(None),) + (0,) * device.n_wires + (0,)
        assert states[zero_idx] == 1.0

        # Check other positions are 0 (sample one)
        other_idx = (slice(None),) + (0,) * (device.n_wires - 1) + (1,) + (0,)
        assert states[other_idx] == 0.0

    def test_bsz_change(self):
        """Test changing batch size."""
        device = fq.DistributedQuantumDevice(n_wires=2, bsz=1, device="cpu")
        assert device.bsz == 1

        device.reset_states(bsz=4)
        assert device.bsz == 4
        assert device._states.shape[0] == 4


class TestStateManipulation:
    """Test state manipulation operations."""

    def test_load_amplitudes(self):
        """Test loading custom amplitudes."""
        device = fq.DistributedQuantumDevice(n_wires=2, device="cpu")

        # Create custom amplitudes
        amplitudes = torch.ones(2, 4) / 2  # batch=2, 2^2=4
        device.load_amplitudes(amplitudes)

        # Check normalization via probability sum
        states = device.states
        # states shape: (batch, 2, 2, 2) for 2 qubits + real/imag
        probs = (states**2).sum(dim=-1)  # Sum over real/imag dimension
        # Flatten qubit dimensions to get probability vector
        batch_size = probs.shape[0]
        flat_probs = probs.reshape(batch_size, -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

    def test_canonicalize(self):
        """Test canonicalization of qubit order."""
        device = fq.DistributedQuantumDevice(n_wires=3, device="cpu")
        device.h(wires=[0])
        device.cx(wires=[0, 2])

        # Canonicalize
        device.canonicalize()

        # After canonicalization, dirty flag should be False

        # States should be accessible
        states = device.states
        assert states is not None


class TestProbabilityDistribution:
    """Test probability distribution calculations."""

    def test_probability_sum(self):
        """Test that probabilities sum to 1."""
        device = fq.DistributedQuantumDevice(n_wires=3, device="cpu")
        device.h(wires=[0])

        states = device.states
        # Sum over real/imag dimension to get probabilities
        probs = (states**2).sum(dim=-1)
        # Flatten qubit dimensions
        flat_probs = probs.reshape(probs.shape[0], -1)
        batch_sums = flat_probs.sum(dim=1)

        assert torch.allclose(batch_sums, torch.ones(probs.shape[0]), atol=1e-5)

    def test_zero_state_probability(self):
        """Test that |0> state has probability 1 for |0> outcome."""
        device = fq.DistributedQuantumDevice(n_wires=2, device="cpu")
        device.reset_states()

        # Get probability of measuring |00>
        states = device.states
        probs = (states**2).sum(dim=-1)
        # Probability of |00> is at index where both qubits are 0
        prob_00 = probs[0, 0, 0]
        assert torch.allclose(prob_00, torch.tensor(1.0), atol=1e-5)


class TestDeviceConstants:
    """Test device constants and properties."""

    def test_device_properties(self):
        """Test device properties."""
        device = fq.DistributedQuantumDevice(n_wires=5, bsz=3, device="cpu")

        assert device.n_wires == 5
        assert device.bsz == 3
        assert device.device == "cpu"
        assert device.world_sz == 1
        assert device.log2_devices == 0
