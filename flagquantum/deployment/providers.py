"""Public aggregation surface for quantum execution providers."""

from ..providers.execution import result_parsing as _result_parsing
from ..providers.execution.braket import (
    AmazonBraketProvider,
    BraketSubmissionPreview,
    braket_backend_profile,
)
from ..providers.execution.cqlib import GuodunProvider, TianyanProvider
from ..providers.execution.fieldquantum import FieldQuantumProvider
from ..providers.execution.http import (
    HttpQuantumProvider,
    ProviderCredentials,
    ProviderEndpoints,
    QuantumCloudTransport,
    UrllibTransport,
)
from ..providers.execution.originq import OriginQProvider
from ..providers.execution.quafu import QuafuProvider
from ..providers.execution.tencent import TencentQuantumProvider

_extract_counts = _result_parsing._extract_counts
_flip_counts = _result_parsing._flip_counts
_format_circuit_source = _result_parsing._format_circuit_source
_normalize_counts = _result_parsing._normalize_counts
_strip_barrier = _result_parsing._strip_barrier
_unwrap_result_envelope = _result_parsing._unwrap_result_envelope

__all__ = [
    "AmazonBraketProvider",
    "BraketSubmissionPreview",
    "FieldQuantumProvider",
    "GuodunProvider",
    "HttpQuantumProvider",
    "OriginQProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "QuafuProvider",
    "QuantumCloudTransport",
    "TencentQuantumProvider",
    "TianyanProvider",
    "UrllibTransport",
    "braket_backend_profile",
]
