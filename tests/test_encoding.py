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

# tests/test_encoding.py
"""Tests for quantum encoding methods."""

import pytest
import torch

import flagquantum as fq


class TestGeneralEncoder:
    """Test GeneralEncoder class."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=4, bsz=2, device="cpu")

    def test_angle_encoder(self, device):
        """Test angle encoding."""
        func_list = [
            {"func": "ry", "wires": [0], "input_idx": 0},
            {"func": "ry", "wires": [1], "input_idx": 1},
        ]
        encoder = fq.GeneralEncoder(func_list)

        x = torch.randn(2, 2)  # batch=2, features=2
        encoder(device, x)

        # Check that state is not all zeros
        states = device.states
        assert states is not None
        assert not torch.allclose(states, torch.zeros_like(states))

        # Check probability sum is 1 for each batch
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(probs.shape[0]), atol=1e-5)

    def test_encoder_with_cx(self, device):
        """Test encoder with non-parameterized gates."""
        func_list = [
            {"func": "ry", "wires": [0], "input_idx": 0},
            {"func": "cx", "wires": [0, 1]},
        ]
        encoder = fq.GeneralEncoder(func_list)

        x = torch.randn(2, 1)
        encoder(device, x)

        # Check that encoder applied successfully (states changed from initial)
        states = device.states
        assert states is not None

        # Initial state would have |0...0> amplitude = 1 at specific position
        # After encoding, it should be distributed
        zero_idx = (slice(None),) + (0,) * device.n_wires + (0,)
        initial_amplitude = states[zero_idx].clone()
        # After encoding, the amplitude at |0...0> should not be 1
        assert not torch.allclose(initial_amplitude, torch.ones_like(initial_amplitude))

    def test_encoder_inverse(self, device):
        """Test encoder inverse."""
        func_list = [
            {"func": "ry", "wires": [0], "input_idx": 0},
            {"func": "ry", "wires": [1], "input_idx": 1},
        ]
        encoder = fq.GeneralEncoder(func_list)

        x = torch.randn(2, 2)

        # Save initial state
        zero_idx = (slice(None),) + (0,) * device.n_wires + (0,)

        # Apply encoding
        encoder(device, x)

        # Apply inverse
        encoder.inverse(device, x)

        # Should return to initial state (|0...0>)
        final_states = device.states
        final_amplitude = final_states[zero_idx]

        # After inverse, should be back to initial (amplitude ~1 at |0...0>)
        assert torch.allclose(
            final_amplitude, torch.ones_like(final_amplitude), atol=1e-5
        )

    def test_angle_encoder_convenience(self, device):
        """Test convenience angle_encoder function."""
        x = torch.randn(2, 4)
        fq.angle_encoder(device, x, wires=[0, 1, 2, 3], rotation="ry")

        # Check that state is not all zeros
        states = device.states
        assert states is not None
        assert not torch.allclose(states, torch.zeros_like(states))

        # Check probability sum is 1
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(probs.shape[0]), atol=1e-5)


class TestAmplitudeEncoding:
    """Test amplitude encoding."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=2, bsz=2, device="cpu")

    def test_amplitude_encoding(self, device):
        """Test amplitude encoding."""
        amplitudes = torch.randn(2, 4)  # batch=2, 2^2=4
        fq.amplitude_encoder(device, amplitudes)

        # Check normalization via probability sum
        states = device.states
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_complex_amplitudes(self, device):
        """Test amplitude encoding with complex amplitudes."""
        amplitudes = torch.randn(2, 4) + 1j * torch.randn(2, 4)
        fq.amplitude_encoder(device, amplitudes)

        # Check normalization via probability sum
        states = device.states
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_amplitude_encoding_shape(self, device):
        """Test amplitude encoding preserves batch dimension."""
        amplitudes = torch.randn(3, 4)  # batch=3
        fq.amplitude_encoder(device, amplitudes)

        states = device.states
        assert states.shape[0] == 3  # batch dimension
        assert device.bsz == 3


class TestEncodingCircuits:
    """Test encoding circuit factory."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=4, bsz=2, device="cpu")

    def test_create_angle_circuit(self, device):
        """Test creating angle encoding circuit."""
        encoder = fq.create_encoding_circuit("angle", n_qubits=4, n_features=4)
        assert isinstance(encoder, fq.GeneralEncoder)

        x = torch.randn(2, 4)
        encoder(device, x)

        # Check that encoding was applied
        states = device.states
        assert states is not None
        assert not torch.allclose(states, torch.zeros_like(states))

        # Check probability sum is 1
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(probs.shape[0]), atol=1e-5)

    def test_create_basis_circuit(self, device):
        """Test creating basis encoding circuit."""
        encoder = fq.create_encoding_circuit("basis", n_qubits=4, n_features=4)
        assert isinstance(encoder, fq.GeneralEncoder)

        x = torch.randint(0, 2, (2, 4))  # binary data
        encoder(device, x)

        # Check that encoding was applied
        states = device.states
        assert states is not None

        # Probability sum should still be 1
        probs = (states**2).sum(dim=-1)
        flat_probs = probs.reshape(probs.shape[0], -1)
        norms = flat_probs.sum(dim=1)
        assert torch.allclose(norms, torch.ones(probs.shape[0]), atol=1e-5)

    def test_create_angle_circuit_different_features(self, device):
        """Test angle circuit with fewer features than qubits."""
        encoder = fq.create_encoding_circuit("angle", n_qubits=6, n_features=3)
        assert isinstance(encoder, fq.GeneralEncoder)

        x = torch.randn(2, 3)
        encoder(device, x)

        # Should work without error
        states = device.states
        assert states is not None
