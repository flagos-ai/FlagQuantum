"""Shared gate and parameter conversions for JAX executors."""

from __future__ import annotations

from typing import Any

from .runtime_environment import (
    _jax_complex_dtype,
    _jax_real_dtype,
    _require_jax,
    _require_torch,
)


def _gate_matrix_as_jax(
    instruction: Any, *, torch_dtype: Any, jax_dtype: Any
) -> tuple[Any, bool]:
    from ....simulation.statevector.operations import _instruction_matrix

    torch = _require_torch()
    _, jnp = _require_jax()
    matrix_torch = _instruction_matrix(
        instruction, device=torch.device("cpu"), dtype=torch_dtype
    )
    diagonal = bool(
        torch.count_nonzero(
            matrix_torch - torch.diag(torch.diagonal(matrix_torch))
        ).item()
        == 0
    )
    matrix = jnp.asarray(matrix_torch.detach().cpu().numpy(), dtype=jax_dtype)
    return matrix, diagonal


def _parameterized_gate_matrix_as_jax(
    instruction: Any, *, complex_bytes: int
) -> tuple[Any, bool]:
    from flagquantum.simulation.numerics.jax import _jax_instruction_matrix

    _, jnp = _require_jax()
    matrix = _jax_instruction_matrix(instruction).astype(
        _jax_complex_dtype(complex_bytes)
    )
    parameterized_diagonal_gates = {
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "rz",
        "phase",
        "u1",
        "cz",
        "crz",
        "cphase",
        "rzz",
    }
    return matrix, str(instruction.name).lower() in parameterized_diagonal_gates


def _jax_parameter_array_from_input(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    jax, jnp = _require_jax()
    torch = _require_torch()
    real_dtype = _jax_real_dtype(complex_bytes)
    if torch.is_tensor(parameters):
        tensor = parameters.detach().contiguous()
        try:
            from jax import dlpack as jax_dlpack

            return jnp.asarray(jax_dlpack.from_dlpack(tensor), dtype=real_dtype)
        except Exception:
            return jnp.asarray(np.asarray(tensor.cpu()).copy(), dtype=real_dtype)
    return jnp.asarray(parameters, dtype=real_dtype)


def _torch_parameters_for_static_build(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    torch = _require_torch()
    if torch.is_tensor(parameters):
        return parameters.detach().cpu()
    dtype = torch.float64 if int(complex_bytes) == 16 else torch.float32
    return torch.as_tensor(
        np.asarray(parameters).copy(), dtype=dtype, device=torch.device("cpu")
    )
