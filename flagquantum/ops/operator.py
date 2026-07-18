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

# ops/operator.py
"""Quantum operator module for parameterized quantum gates.

This module provides a class-based interface for quantum operators, allowing
parameterized gates to be used as trainable PyTorch modules. This is particularly
useful for variational quantum circuits and quantum machine learning.

Example:
    Basic usage:
    >>> from flagquantum.ops import RY, CX
    >>>
    >>> # Create parameterized rotation operator
    >>> ry_op = RY(wires=[0], trainable=True)
    >>>
    >>> # Create non-parameterized CNOT operator
    >>> cx_op = CX(wires=[0, 1])
    >>>
    >>> # Apply operators in forward pass
    >>> output = ry_op(qdev)
    >>> output = cx_op(qdev)
"""

import warnings
from typing import Any, Callable, List, Optional

import torch

from . import functional, matrices


class Op(torch.nn.Module):
    """Base class for quantum operators.

    This class represents a quantum gate as a PyTorch module, making it
    compatible with automatic differentiation and optimization frameworks.
    It supports both parameterized and non-parameterized gates, with optional
    trainable parameters.

    Important:
        Once an operator is applied to a quantum device (without explicit
        `params` argument), its parameters are initialized with the device's
        batch size. The same operator instance cannot be used with devices of
        different batch sizes when relying on internal parameters. To use with
        different batch sizes, either:
        1. Pass `params` explicitly with matching batch size, or
        2. Create a new operator instance for the new batch size.

    Attributes:
        name: Name of the operator (e.g., 'RY', 'CX').
        func_: Forward function for the gate (e.g., `functional.ry`).
        func_inv_: Inverse function for the gate (e.g., `functional.ry_inv`).
        wires: Qubit indices the operator acts on.
        has_params: Whether the gate has parameters (e.g., rotation angle).
        trainable: Whether the parameters should be trainable (requires_grad).
        _params: The parameter tensor(s) for the gate (lazy initialized).

    Example:
        Creating a trainable rotation operator:
        >>> op = Op(
        ...     name='RY',
        ...     func=functional.ry,
        ...     func_inv=functional.ry_inv,
        ...     wires=[0],
        ...     has_params=True,
        ...     trainable=True,
        ...     init_scale=0.1
        ... )
        >>>
        >>> # Apply operator
        >>> op(qdev)  # Uses internal parameters
        >>> op(qdev, params=torch.tensor([0.5]))  # Override parameters
        >>>
        >>> # Apply inverse
        >>> op.inverse(qdev)
    """

    def __init__(
        self,
        func: Callable,
        func_inv: Callable,
        wires: List[int],
        name: Optional[str] = None,
        has_params: bool = True,
        trainable: bool = True,
        init_scale: float = 1.0,
        init_params: Optional[torch.Tensor] = None,
        n_params: Optional[int] = None,
        **unused: Any,
    ):
        """Initializes a quantum operator.

        Args:
            func: Forward function for the gate (e.g., `functional.ry`).
            func_inv: Inverse function for the gate (e.g., `functional.ry_inv`).
            wires: Qubit indices the operator acts on.
            name: Name of the operator. If None, inferred from func.
            has_params: Whether the gate has parameters. Defaults to True.
            trainable: Whether parameters should be trainable. Defaults to True.
            init_scale: Scale factor for parameter initialization. Defaults to 1.0.
            init_params: Optional initial parameter values. Shape can be
                [n_params] or [batch_size, n_params]. If batch_size=1, it will
                be expanded to match qdev.bsz on first use.
            n_params: Number of parameters for this gate. If None, inferred
                from function signature (for known gates) or defaults to 1.
            **unused: Additional unused keyword arguments (for compatibility).

        Example:
            >>> # Fixed gate (no parameters)
            >>> hadamard = Op(
            ...     name='H',
            ...     func=functional.h,
            ...     func_inv=functional.h_inv,
            ...     wires=[0],
            ...     has_params=False
            ... )
            >>>
            >>> # Trainable rotation with small initial values
            >>> rx = Op(
            ...     name='RX',
            ...     func=functional.rx,
            ...     func_inv=functional.rx_inv,
            ...     wires=[0],
            ...     trainable=True,
            ...     init_scale=0.01
            ... )
        """
        super().__init__()
        self.name = name
        self.func_ = func
        self.func_inv_ = func_inv
        self.wires = wires
        self.has_params = has_params
        self.trainable = trainable
        self.init_scale = init_scale
        self.init_params = init_params
        self.params: Optional[torch.Tensor] = None  # Lazy initialized

        if n_params is not None:
            self.n_params = n_params
        else:
            self.n_params = self._infer_num_params()

        self._initialized_bsz: Optional[int] = (
            None  # Track which batch size we initialized for
        )

    def _infer_num_params(self) -> int:
        """Infer number of parameters from function signature or name.

        Returns:
            Number of parameters expected by the gate.
        """
        if not self.has_params:
            return 0

        # Try to infer from function name

        # Known multi-parameter gates
        if self.name == "U2":
            return 2
        elif self.name == "U3":
            return 3

        # Default to single parameter for rotation gates
        return 1

    def _prepare_init_params(self, init_params: torch.Tensor, bsz: int) -> torch.Tensor:
        """Prepare and validate initial parameters.

        Args:
            init_params: User-provided initial parameters.
            bsz: Target batch size.

        Returns:
            Properly shaped parameter tensor.
        """
        if not isinstance(init_params, torch.Tensor):
            init_params = torch.tensor(init_params, dtype=torch.float32)

        # Handle scalar case (0-dimensional tensor)
        if init_params.dim() == 0:
            init_params = init_params.reshape(1, 1)

        # Shape: [n_params] -> [1, n_params]
        if init_params.dim() == 1:
            init_params = init_params.unsqueeze(0)

        # Expand single batch to target batch size
        if init_params.shape[0] == 1 and bsz > 1:
            init_params = init_params.expand(bsz, -1)

        # Validate shape
        expected_shape = (bsz, self.n_params)
        if init_params.shape != expected_shape:
            raise ValueError(
                f"Gate {self.name}: init_params shape {init_params.shape} doesn't match expected "
                f"shape {expected_shape} (batch_size={bsz}, n_params={self.n_params})"
            )

        return init_params

    def _initialize_params(self, bsz: int) -> torch.Tensor:
        """Initialize parameters for a given batch size.

        Args:
            bsz: Batch size from the quantum device.

        Returns:
            The parameter tensor (as nn.Parameter if trainable).
        """
        if self.init_params is not None:
            params = self._prepare_init_params(self.init_params, bsz)
        else:
            # Random initialization: uniform in [-init_scale/2, init_scale/2]
            params = self.init_scale * (torch.rand(bsz, self.n_params) - 0.5)

        # Wrap as nn.Parameter if trainable
        if self.trainable:
            params = torch.nn.Parameter(params)

        self.params = params
        self._initialized_bsz = bsz
        return self.params

    def _get_params(
        self, qdev, params_override: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Get parameters for application.

        Args:
            qdev: Quantum device instance.
            params_override: User-provided parameter override.

        Returns:
            Parameters to use, or None if gate has no parameters.

        Raises:
            ValueError: If batch size mismatch and no override provided.
        """
        # User provided params - validate and normalize shape
        if params_override is not None:
            expected_bsz = qdev.bsz
            expected_n_params = self.n_params
            normalized_params = params_override

            # Handle scalar case (0-dimensional)
            if normalized_params.ndim == 0:
                if expected_bsz == 1 and expected_n_params == 1:
                    normalized_params = normalized_params.reshape(1, 1)
                else:
                    raise ValueError(
                        f"Gate {self.name}: Scalar parameter provided but expected "
                        f"shape ({expected_bsz}, {expected_n_params}). "
                        f"Please provide a tensor with shape ({expected_bsz}, {expected_n_params})"
                    )

            # Handle 1D case: could be [n_params] for bsz=1 or [bsz] for n_params=1
            elif normalized_params.ndim == 1:
                if (
                    expected_bsz == 1
                    and normalized_params.shape[0] == expected_n_params
                ):
                    # Case: [phi, lam] for U2 gate with batch size 1
                    normalized_params = normalized_params.reshape(1, expected_n_params)
                elif (
                    expected_n_params == 1
                    and normalized_params.shape[0] == expected_bsz
                ):
                    # Case: [theta1, theta2, ...] for single-parameter gates
                    normalized_params = normalized_params.reshape(expected_bsz, 1)
                else:
                    raise ValueError(
                        f"Gate {self.name}: Provided 1D params shape {tuple(params_override.shape)} "
                        f"is invalid. Expected shape ({expected_bsz}, {expected_n_params})"
                        + (
                            f" or ({expected_n_params},) for batch size 1."
                            if expected_bsz == 1
                            else ""
                        )
                        + (
                            f" or ({expected_bsz},) for single-parameter gates."
                            if expected_n_params == 1
                            else ""
                        )
                    )

            # Handle 2D case: should match exactly the expected shape
            elif normalized_params.ndim == 2:
                if normalized_params.shape != (expected_bsz, expected_n_params):
                    raise ValueError(
                        f"Gate {self.name}: Provided params shape {tuple(params_override.shape)} "
                        f"doesn't match expected shape ({expected_bsz}, {expected_n_params})"
                    )

            # Handle higher dimensions
            else:
                raise ValueError(
                    f"Gate {self.name}: Provided params shape {tuple(params_override.shape)} "
                    f"is invalid (maximum 2 dimensions allowed). Expected shape "
                    f"({expected_bsz}, {expected_n_params})"
                )

            # Validate for gates with unknown n_params (e.g., custom gates)
            if expected_n_params is None:
                # For gates without predefined parameter count (like custom gates),
                # we just ensure batch dimension matches
                if normalized_params.ndim == 0:
                    if expected_bsz != 1:
                        raise ValueError(
                            f"Gate {self.name}: Provided scalar parameter but batch size "
                            f"is {expected_bsz}. Please provide a tensor with first dimension {expected_bsz}"
                        )
                    normalized_params = normalized_params.reshape(1)
                elif normalized_params.shape[0] != expected_bsz:
                    raise ValueError(
                        f"Gate {self.name}: Provided params batch size ({normalized_params.shape[0]}) "
                        f"doesn't match qdev batch size ({expected_bsz})"
                    )

            self.params = normalized_params
            return self.params

        # No parameters for this gate
        if not self.has_params:
            if self.params is not None:
                warnings.warn(
                    f"Gate {self.name} has has_params=False but "
                    f"internal parameters exist. Ignoring parameters.",
                    UserWarning,
                )
            return None

        # Use internal parameters - must be initialized
        if self.params is None:
            # First time: initialize with current batch size
            return self._initialize_params(qdev.bsz)

        return self.params

    def _apply_op(
        self,
        qdev,
        func: Callable,
        wires: Optional[List[int]],
        params: Optional[torch.Tensor],
    ) -> None:
        """Apply an operator (forward or inverse) to the quantum device.

        Args:
            qdev: Quantum device instance.
            func: Function to apply (func_ or func_inv_).
            wires: Optional wire override.
            params: Optional parameter override.
        """
        final_params = self._get_params(qdev, params)

        func(
            qdev,
            wires if wires is not None else self.wires,
            params=final_params,
        )

    def forward(
        self,
        qdev,
        wires: Optional[List[int]] = None,
        params: Optional[torch.Tensor] = None,
    ):
        """Applies the operator to the quantum device.

        Args:
            qdev: The quantum device instance (e.g., DistributedQuantumDevice).
            wires: Qubit indices to apply the gate on. Overrides the instance's
                wires if provided. Defaults to None.
            params: Parameter values for the gate. If provided, must have shape
                (qdev.bsz, n_params). Overrides internal parameters.

        Returns:
            The result of applying the operator (typically the modified qdev).

        Note:
            When using internal parameters (not passing `params`), the operator
            instance is locked to the batch size of the first qdev it's applied to.
            To use with different batch sizes, either:
            1. Pass `params` explicitly with matching batch size, or
            2. Create a new operator instance for the new batch size.

        Example:
            >>> ry_op = RY(wires=[0], trainable=True)
            >>>
            >>> # Use default wires and learned parameters
            >>> ry_op(qdev)
            >>>
            >>> # Override wires
            >>> ry_op(qdev, wires=[2])
            >>>
            >>> # Override parameters
            >>> ry_op(qdev, params=torch.tensor([3.14159]))
            >>>
            >>> # Use with different batch size (explicit params)
            >>> params_64 = torch.randn(64, 1)
            >>> ry_op(qdev_64, params=params_64)
        """
        self._apply_op(qdev, self.func_, wires, params)

    def inverse(
        self,
        qdev,
        wires: Optional[List[int]] = None,
        params: Optional[torch.Tensor] = None,
    ):
        """Applies the inverse of the operator to the quantum device.

        Args:
            qdev: The quantum device instance.
            wires: Qubit indices to apply the gate on. Overrides the instance's
                wires if provided. Defaults to None.
            params: Parameter values for the gate. Overrides the instance's
                parameters if provided. Defaults to None.

        Returns:
            The result of applying the inverse operator.

        Example:
            >>> ry_op = RY(wires=[0], trainable=True)
            >>>
            >>> # Apply forward then inverse (should return to original state)
            >>> ry_op(qdev, params=torch.tensor([0.5]))
            >>> ry_op.inverse(qdev, params=torch.tensor([0.5]))
            >>>
            >>> # Using different parameters for inverse
            >>> ry_op(qdev, params=torch.tensor([0.5]))
            >>> ry_op.inverse(qdev, params=torch.tensor([-0.5]))
        """
        self._apply_op(qdev, self.func_inv_, wires, params)

    def __repr__(self) -> str:
        """Return a string representation of the operator."""
        params_info = f", params={self.params}" if self.params is not None else ""
        return f"{self.name}(wires={self.wires}, has_params={self.has_params}{params_info})"


def op_factory(
    name: str,
    module,
    has_params: bool = True,
    trainable: bool = True,
) -> type:
    """Factory that dynamically creates operator classes.

    This factory function creates a new operator class for a given gate name.
    The generated class is named in uppercase (e.g., 'RY' from 'ry') and inherits
    from the base Op class.

    Args:
        name: Name of the gate in lowercase (e.g., 'ry', 'cx', 'h').
        module: Module containing the gate functions (e.g., functional module).
        has_params: Default value for whether the gate has parameters.
            Defaults to True.
        trainable: Default value for whether parameters are trainable.
            Defaults to True.

    Returns:
        A new operator class (subclass of Op) for the specified gate.

    Example:
        Creating an RY operator class:
        >>> RY = op_factory('ry', functional, has_params=True, trainable=True)
        >>> ry_op = RY(wires=[0])  # Creates an RY operator instance
        >>> ry_op(qdev, params=torch.tensor([0.5]))

        Creating a fixed gate class (no parameters):
        >>> H = op_factory('h', functional, has_params=False, trainable=False)
        >>> h_op = H(wires=[0])
        >>> h_op(qdev)  # Applies Hadamard gate

        Using in a variational circuit:
        >>> class VariationalCircuit(torch.nn.Module):
        ...     def __init__(self, n_qubits):
        ...         super().__init__()
        ...         self.ry1 = RY(wires=[0], trainable=True)
        ...         self.ry2 = RY(wires=[1], trainable=True)
        ...         self.cx = CX(wires=[0, 1])
        ...
        ...     def forward(self, qdev):
        ...         self.ry1(qdev)
        ...         self.ry2(qdev)
        ...         self.cx(qdev)
        ...         return qdev.states
    """

    def init_func(self, wires: List[int], **kwargs):
        """Initialize the operator instance.

        Args:
            wires: Qubit indices the operator acts on.
            **kwargs: Additional arguments passed to Op constructor.
                Common arguments include:
                - has_params: Override default has_params
                - trainable: Override default trainable
                - init_scale: Initialization scale for parameters
                - init_params: Initial parameter values
        """
        # Merge defaults with user overrides
        kwargs.setdefault("has_params", has_params)
        kwargs.setdefault("trainable", trainable)
        kwargs.setdefault("name", name.upper())  # 设置 operator name

        Op.__init__(
            self,
            getattr(module, name),
            getattr(module, f"{name}_inv"),
            wires,
            **kwargs,
        )

    newclass = type(name.upper(), (Op,), {"__init__": init_func})
    return newclass


# Dynamically create operator classes for all built-in gates
for name_, mat_ in matrices.GATE_MAT_DICT.items():
    kw = {} if callable(mat_) else {"has_params": False, "trainable": False}
    vars()[name_.upper()] = op_factory(name_, functional, **kw)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "Op",
    "op_factory",
] + [name.upper() for name in matrices.GATE_MAT_DICT.keys()]
