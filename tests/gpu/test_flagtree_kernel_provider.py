"""Real-device identity checks for the FlagTree compiler substitution."""

from importlib import import_module, metadata

import pytest
import torch

from flagquantum.runtime.executors.statevector.kernel_dispatch import (
    select_triton_kernel,
    triton_compiler_provenance,
)

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def test_flagtree_owns_the_triton_namespace_on_cuda() -> None:
    triton = import_module("triton")
    owners = tuple(
        sorted(
            {
                distribution.casefold().replace("_", "-")
                for distribution in metadata.packages_distributions().get("triton", ())
            }
        )
    )

    assert owners == ("flagtree",)
    assert metadata.version("flagtree")
    assert triton.__version__
    assert torch.cuda.is_available()
    assert triton.runtime.driver.active.get_current_target().backend == "cuda"


def test_dispatch_reports_the_measured_flagtree_route() -> None:
    distribution, version, integration_path, status = triton_compiler_provenance()

    assert distribution == "flagtree"
    assert version == metadata.version("flagtree")
    assert integration_path == "flagtree"
    assert status == "resolved"

    decision = select_triton_kernel(
        "flagtree_provider_probe",
        requested=True,
        available=True,
        device_runtime_provider="pytorch",
        device_type="cuda",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )
    summary = decision.summary()

    assert summary["kernel_compiler"] == {
        "distribution": "flagtree",
        "version": version,
        "backend": "cuda",
        "identity_source": "python_package_metadata",
        "identity_status": "resolved",
    }
    assert summary["kernel_route"]["integration_path"] == "flagtree"
    assert summary["kernel_route"]["fallback"] is False
