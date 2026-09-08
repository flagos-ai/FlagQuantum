"""Capability discovery for solver-native MPS factorization workspaces."""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import asdict, dataclass

import torch

_REQUIRED_XGESVD_SYMBOLS = (
    "cusolverDnCreateParams",
    "cusolverDnDestroyParams",
    "cusolverDnXgesvd_bufferSize",
    "cusolverDnXgesvd",
)
_REQUIRED_CLASSIC_GESVD_SYMBOLS = (
    "cusolverDnCgesvd_bufferSize",
    "cusolverDnCgesvd",
    "cusolverDnZgesvd_bufferSize",
    "cusolverDnZgesvd",
)


@dataclass(frozen=True)
class SolverWorkspaceCapabilities:
    """Auditable boundary between PyTorch and native cuSOLVER workspaces."""

    torch_version: str
    torch_cuda_version: str | None
    torch_public_external_workspace: bool
    cusolver_library: str | None
    cusolver_loadable: bool
    classic_complex_gesvd_workspace_api: bool
    xgesvd_device_host_workspace_api: bool
    native_extension_required: bool
    recommended_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _has_symbols(library: ctypes.CDLL, symbols: tuple[str, ...]) -> bool:
    return all(hasattr(library, symbol) for symbol in symbols)


def probe_solver_workspace_capabilities() -> SolverWorkspaceCapabilities:
    """Inspect symbols without invoking cuSOLVER or depending on a CUDA context."""

    library_name = ctypes.util.find_library("cusolver")
    library = None
    if library_name:
        try:
            library = ctypes.CDLL(library_name)
        except OSError:
            library = None
    classic = (
        _has_symbols(library, _REQUIRED_CLASSIC_GESVD_SYMBOLS)
        if library is not None
        else False
    )
    xgesvd = (
        _has_symbols(library, _REQUIRED_XGESVD_SYMBOLS)
        if library is not None
        else False
    )
    # torch.linalg.svd exposes driver selection but no caller-owned device or
    # host workspace parameter. Do not infer support from private ATen symbols.
    torch_public_workspace = False
    native_required = (classic or xgesvd) and not torch_public_workspace
    if xgesvd:
        recommendation = "cpp_cuda_extension_cusolver_xgesvd"
    elif classic:
        recommendation = "cpp_cuda_extension_classic_gesvd"
    else:
        recommendation = "torch_linalg_internal_workspace"
    return SolverWorkspaceCapabilities(
        torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda,
        torch_public_external_workspace=torch_public_workspace,
        cusolver_library=library_name,
        cusolver_loadable=library is not None,
        classic_complex_gesvd_workspace_api=classic,
        xgesvd_device_host_workspace_api=xgesvd,
        native_extension_required=native_required,
        recommended_path=recommendation,
    )


__all__ = (
    "SolverWorkspaceCapabilities",
    "probe_solver_workspace_capabilities",
)
