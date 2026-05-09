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

from typing import Any, Callable, List, Optional

import torch

from . import functional, matrices


class Op(torch.nn.Module):
    """Base class for quantum operators.

    This class represents a quantum gate as a PyTorch module, making it
    compatible with automatic differentiation and optimization frameworks.
    It supports both parameterized and non-parameterized gates, with optional
    trainable parameters.

    Attributes:
        func_: Forward function for the gate (e.g., `functional.ry`).
        func_inv_: Inverse function for the gate (e.g., `functional.ry_inv`).
        wires: Qubit indices the operator acts on.
        has_params: Whether the gate has parameters (e.g., rotation angle).
        trainable: Whether the parameters should be trainable (requires_grad).
        params: The parameter tensor(s) for the gate.

    Example:
        Creating a trainable rotation operator:
        >>> op = Op(
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
        has_params: bool = True,
        trainable: bool = True,
        init_scale: float = 1,
        **unused: Any,
    ):
        """Initializes a quantum operator.

        Args:
            func: Forward function for the gate (e.g., `functional.ry`).
            func_inv: Inverse function for the gate (e.g., `functional.ry_inv`).
            wires: Qubit indices the operator acts on.
            has_params: Whether the gate has parameters. Defaults to True.
            trainable: Whether parameters should be trainable. Defaults to True.
            init_scale: Scale factor for parameter initialization. Defaults to 1.
            **unused: Additional unused keyword arguments (for compatibility).

        Example:
            >>> # Fixed gate (no parameters)
            >>> hadamard = Op(
            ...     func=functional.h,
            ...     func_inv=functional.h_inv,
            ...     wires=[0],
            ...     has_params=False
            ... )
            >>>
            >>> # Trainable rotation with small initial values
            >>> rx = Op(
            ...     func=functional.rx,
            ...     func_inv=functional.rx_inv,
            ...     wires=[0],
            ...     trainable=True,
            ...     init_scale=0.01
            ... )
        """
        super().__init__()
        self.func_ = func
        self.func_inv_ = func_inv
        self.wires = wires
        self.has_params = has_params
        self.trainable = trainable
        self.params = None

        if has_params:
            # Initialize parameter randomly between -0.5 and 0.5, scaled by init_scale
            self.params = init_scale * (torch.rand(1) - 0.5)
            if trainable:
                self.params = torch.nn.Parameter(self.params)

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
            params: Parameter values for the gate. Overrides the instance's
                parameters if provided. Defaults to None.

        Returns:
            The result of applying the operator (typically the modified qdev).

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
        """
        self.func_(
            qdev,
            wires if wires is not None else self.wires,
            params=params if params is not None else self.params,
        )

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
        self.func_inv_(
            qdev,
            wires if wires is not None else self.wires,
            params=params if params is not None else self.params,
        )


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
        """
        kwargs.update({"has_params": kwargs.get("has_params", has_params)})
        kwargs.update({"trainable": kwargs.get("trainable", trainable)})
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
