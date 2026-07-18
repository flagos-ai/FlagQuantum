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

# tests/test_gates.py
"""Tests for quantum gates."""

import pytest
import torch

import flagquantum as fq


class TestSingleQubitGates:
    """Test single-qubit gates."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=2, bsz=1, device="cpu")

    def test_hadamard(self, device):
        """Test Hadamard gate using measurement."""
        device.reset_states()
        device.h(wires=[0])

        # H|0> = |+>, expectation of Z should be 0
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0]) < 1e-5

    def test_pauli_x(self, device):
        """Test Pauli X gate using double application."""
        device.reset_states()
        device.x(wires=[0])
        # X|0> = |1>, apply X again should return to |0>
        device.x(wires=[0])

        # Should be back to |0>, Z expectation should be 1
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] - 1.0) < 1e-5

    def test_pauli_z(self, device):
        """Test Pauli Z gate."""
        device.reset_states()
        device.h(wires=[0])  # Create |+>
        device.z(wires=[0])  # Z|+> = |->
        device.h(wires=[0])  # H|-> = |1>

        # Should be in |1> state
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] + 1.0) < 1e-5  # Z expectation = -1 for |1>

    def test_rx_rotation(self, device):
        """Test RX rotation gate."""
        device.reset_states()
        theta = torch.tensor(3.14159)  # π
        device.rx(wires=[0], params=theta)

        # RX(π)|0> = -i|1>, Z expectation should be -1
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] + 1.0) < 1e-5

    def test_ry_rotation(self, device):
        """Test RY rotation gate."""
        device.reset_states()
        theta = torch.tensor(3.14159)  # π
        device.ry(wires=[0], params=theta)

        # RY(π)|0> = |1>, Z expectation should be -1
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] + 1.0) < 1e-5

    def test_x_gate_identity(self, device):
        """Test X gate applied twice is identity."""
        device.reset_states()

        # Save initial amplitudes
        states_before = device.states.clone()
        zero_idx = (slice(None),) + (0,) * device.n_wires + (0,)
        initial_magnitude = states_before[zero_idx].clone()

        # Apply X twice
        device.x(wires=[0])
        device.x(wires=[0])

        # Should be back to initial
        states_after = device.states
        final_magnitude = states_after[zero_idx]
        assert torch.allclose(initial_magnitude, final_magnitude)


class TestTwoQubitGates:
    """Test two-qubit gates."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=2, bsz=1, device="cpu")

    def test_cnot(self, device):
        """Test CNOT gate using measurement correlation."""
        device.reset_states()
        device.x(wires=[0])  # Set control to |1>
        device.cx(wires=[0, 1])  # Apply CNOT

        # Both should be in |1> state
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] + 1.0) < 1e-5  # Qubit 0 in |1>
        assert torch.abs(result[0, 1] + 1.0) < 1e-5  # Qubit 1 in |1>

    def test_cnot_control_0(self, device):
        """Test CNOT with control=0 does nothing."""
        device.reset_states()
        device.x(wires=[1])  # Set target to |1>
        device.cx(wires=[0, 1])  # CNOT with control=0

        # Target should stay |1>
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] - 1.0) < 1e-5  # Qubit 0 in |0>
        assert torch.abs(result[0, 1] + 1.0) < 1e-5  # Qubit 1 in |1>

    def test_swap(self, device):
        """Test SWAP gate."""
        device.reset_states()
        device.x(wires=[0])  # |10>
        device.swap(wires=[0, 1])  # SWAP

        # Should become |01>
        result = fq.measure_allZ(device)
        assert torch.abs(result[0, 0] - 1.0) < 1e-5  # Qubit 0 in |0>
        assert torch.abs(result[0, 1] + 1.0) < 1e-5  # Qubit 1 in |1>

    def test_bell_state(self, device):
        """Test Bell state: individual Z expectations should be zero."""
        device.reset_states()
        device.h(wires=[0])
        device.cx(wires=[0, 1])

        # The expected values ⟨Z₀⟩ and ⟨Z₁⟩ should be close to 0
        exp_vals = fq.measure_allZ(device)  # 不需要 shots
        assert torch.abs(exp_vals[0, 0]) < 1e-5
        assert torch.abs(exp_vals[0, 1]) < 1e-5


class TestParameterizedGates:
    """Test parameterized gates with gradients."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(
            n_wires=1, bsz=1, device="cpu", invertible=True
        )

    def test_parameter_gradient(self, device):
        """Test that parameters have gradients."""
        device.reset_states()

        # Create trainable parameter (ensure it's 1D for scalar output)
        theta = torch.tensor([0.5], requires_grad=True)
        device.rx(wires=[0], params=theta)

        # Measure and take sum to get scalar
        result = fq.measure_allZ(device)
        loss = result.abs().sum()  # Now scalar

        loss.backward()

        assert theta.grad is not None
        assert not torch.isnan(theta.grad)

    def test_parameter_value_change(self, device):
        """Test that parameter value affects measurement outcome."""
        device.reset_states()

        # Small rotation
        theta_small = torch.tensor([0.1])
        device.rx(wires=[0], params=theta_small)
        result_small = fq.measure_allZ(device)

        device.reset_states()

        # Large rotation (π)
        theta_large = torch.tensor([3.14159])
        device.rx(wires=[0], params=theta_large)
        result_large = fq.measure_allZ(device)

        # Different parameters should give different results
        assert not torch.allclose(result_small, result_large, atol=1e-5)

    def test_inverse_gate(self, device):
        """Test inverse gates using sequential application."""
        device.reset_states()
        theta = torch.tensor([0.5])

        # Apply and then invert
        device.rx(wires=[0], params=theta)

        # Apply inverse using multiple methods if rx_inv not available
        try:
            device.rx_inv(wires=[0], params=theta)
        except AttributeError:
            # If rx_inv doesn't exist, use rx with negative parameter
            device.rx(wires=[0], params=-theta)

        # Should return to initial |0> state
        result_after_inv = fq.measure_allZ(device)

        # Initial |0> has Z expectation 1
        # After inverse, should be close to 1
        assert torch.abs(result_after_inv[0, 0] - 1.0) < 1e-5

    def test_rz_gate(self, device):
        """Test RZ rotation gate."""
        device.reset_states()
        device.h(wires=[0])  # Create |+>

        theta = torch.tensor([3.14159])  # π
        device.rz(wires=[0], params=theta)  # Add phase

        device.h(wires=[0])  # Transform back

        # RZ(π) on |+> should give |->
        result = fq.measure_allZ(device)
        # Z expectation should be -1
        assert torch.abs(result[0, 0] + 1.0) < 1e-5


class TestGateCommutations:
    """Test gate commutation relations."""

    @pytest.fixture
    def device(self):
        return fq.DistributedQuantumDevice(n_wires=2, bsz=1, device="cpu")

    def test_x_and_z_anticommute(self, device):
        """Test that X and Z anti-commute."""
        device.reset_states()

        # Apply X then Z
        device.x(wires=[0])
        device.z(wires=[0])
        result_xz = fq.measure_allZ(device)

        device.reset_states()

        # Apply Z then X
        device.z(wires=[0])
        device.x(wires=[0])
        result_zx = fq.measure_allZ(device)

        # XZ|0> = -i|1>, ZX|0> = i|1>, magnitudes same but phases differ
        # Measurement outcomes should be same (both |1>)
        assert torch.abs(result_xz[0, 0] + 1.0) < 1e-5
        assert torch.abs(result_zx[0, 0] + 1.0) < 1e-5
