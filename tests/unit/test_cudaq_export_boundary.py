from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.parameters import Parameter
from flagquantum.ecosystem import available_adapters, get_adapter
from flagquantum.ecosystem.cudaq import (
    CudaqConversionError,
    CudaqDependencyError,
    _version,
    conversion,
    export_cudaq,
)

pytestmark = pytest.mark.unit


class _Kernel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def qalloc(self, count: int) -> tuple[int, ...]:
        return tuple(range(count))

    def __getattr__(self, name: str):
        def append(*args: object) -> None:
            self.calls.append((name, *args))

        return append


def _sdk(kernel: _Kernel) -> SimpleNamespace:
    return SimpleNamespace(__version__="0.16.0.post1", make_kernel=lambda: kernel)


def test_namespace_and_registry_are_lazy() -> None:
    sys.modules.pop("cudaq", None)
    assert "cudaq" in available_adapters()
    assert get_adapter("cudaq").name == "cudaq"
    assert "cudaq" not in sys.modules


def test_export_builds_supported_kernel_in_ir_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _Kernel()
    monkeypatch.setattr(conversion, "import_module", lambda name: _sdk(kernel))
    result = export_cudaq(Circuit(2).h(0).ry(1, theta=0.25).cx(0, 1))
    assert result.kernel is kernel
    assert kernel.calls == [("h", 0), ("ry", 0.25, 1), ("cx", 0, 1)]
    assert result.report.lossless
    assert result.report.framework_version == "0.16.0.post1"


def test_export_reports_distribution_version_instead_of_verbose_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _Kernel()
    sdk = _sdk(kernel)
    sdk.__version__ = "CUDA-Q Version 0.15.1 (https://github.com/NVIDIA/cuda-quantum)"
    monkeypatch.setattr(conversion, "import_module", lambda name: sdk)
    monkeypatch.setattr(_version.metadata, "version", lambda name: "0.15.1")

    result = export_cudaq(Circuit(1).x(0))

    assert result.report.framework_version == "0.15.1"


def test_missing_dependency_fails_only_when_export_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(conversion, "import_module", missing)
    with pytest.raises(CudaqDependencyError, match=r"flagquantum\[cudaq\]"):
        export_cudaq(Circuit(1).x(0))


@pytest.mark.parametrize(
    ("ir", "code"),
    [
        (CircuitIR(1, (Instruction("i", (0,)),)), "unsupported_operation"),
        (
            CircuitIR(1, (Instruction("rx", (0,), {"theta": Parameter("t")}),)),
            "symbolic_parameter_not_supported",
        ),
        (
            CircuitIR(1, (Instruction("rx", (0,), {"theta": float("inf")}),)),
            "non_finite_parameter",
        ),
        (
            CircuitIR(1, (), measurements=(MeasurementNode("m", (0,)),)),
            "measurement_not_supported",
        ),
    ],
)
def test_unsupported_semantics_fail_before_kernel_construction(
    monkeypatch: pytest.MonkeyPatch, ir: CircuitIR, code: str
) -> None:
    kernel = _Kernel()
    monkeypatch.setattr(conversion, "import_module", lambda name: _sdk(kernel))
    with pytest.raises(CudaqConversionError) as caught:
        export_cudaq(ir, allow_lossy=True)
    assert code in {issue.code for issue in caught.value.report.issues}
    assert kernel.calls == []


def test_reverse_conversion_is_explicitly_rejected() -> None:
    with pytest.raises(CudaqConversionError) as caught:
        get_adapter("cudaq").import_program(object())
    assert caught.value.report.issues[0].code == "reverse_conversion_not_supported"
