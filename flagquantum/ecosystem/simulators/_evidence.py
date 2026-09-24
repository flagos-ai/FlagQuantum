"""Packaged projection of the checked-in simulator comparison evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SOURCE_PATH = (
    "benchmarks/results/comparison/simulator_comparison_cpu_arm64_20260923.json"
)
SOURCE_SHA256 = "e2cc3d087a620b1c9dcf08401eb79a915ac39d156fc23da29c9ff532c407bb06"
WORKLOAD_MANIFEST_PATH = "benchmarks/manifests/simulator_workload_ir_v1.json"
WORKLOAD_MANIFEST_SHA256 = (
    "0e8d5ec9797b43c65e1c668abc90e7190e93510ef6633f0c9c744dfa7a0147d0"
)

_IDENTITY = {
    "calls_per_sample": 10,
    "compilation_included_in_steady_state": False,
    "conversion_included_in_steady_state": False,
    "device": "cpu",
    "hidden_fallback_allowed": False,
    "iterations": 9,
    "machine": "arm64",
    "output_basis_normalization_included_in_execution": False,
    "platform": "macOS-27.0-arm64-arm-64bit",
    "processor": "arm",
    "python": "3.12.14",
    "result_retrieval_included_in_execution": True,
    "setup_iterations": 3,
    "torch": "2.13.0",
    "torch_threads": 1,
    "warmup": 3,
    "world_size": 1,
}

_ROWS = (
    (
        10,
        82,
        "649aeba239b1fd06d25eb5eaf3637ecd5d962943dd6c4b22c051b19abd7d42b2",
        "255726a8a31f9b38cdc12674422b476a96c119ebae4303b8389848b2fd3796f9",
        {
            "flagquantum_native": (
                "FlagQuantum",
                "0.2.0",
                0.0005772208038251847,
                0.020305563272991507,
                True,
            ),
            "qiskit_aer": (
                "Qiskit Aer",
                "0.17.2",
                0.0006682250008452683,
                0.02368842955047254,
                True,
            ),
            "cirq_simulator": (
                "Cirq Simulator",
                "1.7.0",
                0.006519104196922853,
                0.2961993148743698,
                False,
            ),
            "pennylane_lightning_qubit": (
                "PennyLane Lightning",
                "0.45.1",
                0.000554479198763147,
                0.03634041121302801,
                True,
            ),
        },
    ),
    (
        14,
        114,
        "b0989ab5f38dd830e03accbe16ed8cf6dad3c2d1b78fc606d9e001dc182b43aa",
        "9e694b13fd7189e728b9ac688dcdf489c5eac9d1653c4148129ddd1b95080121",
        {
            "flagquantum_native": (
                "FlagQuantum",
                "0.2.0",
                0.0013067167019471526,
                0.047861636363166614,
                True,
            ),
            "qiskit_aer": (
                "Qiskit Aer",
                "0.17.2",
                0.0024814500007778405,
                0.019907716702515742,
                True,
            ),
            "cirq_simulator": (
                "Cirq Simulator",
                "1.7.0",
                0.017390829196665437,
                0.10592627765981084,
                False,
            ),
            "pennylane_lightning_qubit": (
                "PennyLane Lightning",
                "0.45.1",
                0.001487791701219976,
                0.004721763872698544,
                True,
            ),
        },
    ),
    (
        18,
        146,
        "7679c254999c4522f9f99090c2a8ce1ad6fc64eab887758617b08986a09167f4",
        "ec19f1ff0bcab30ea7a9db81f03c2b501d5d8d93af774f27739ef59974dc72de",
        {
            "flagquantum_native": (
                "FlagQuantum",
                "0.2.0",
                0.013124554097885266,
                0.005950366204597104,
                True,
            ),
            "qiskit_aer": (
                "Qiskit Aer",
                "0.17.2",
                0.02687223330140114,
                0.0017260678689264826,
                True,
            ),
            "cirq_simulator": (
                "Cirq Simulator",
                "1.7.0",
                0.06917833330226131,
                0.24380947754053467,
                False,
            ),
            "pennylane_lightning_qubit": (
                "PennyLane Lightning",
                "0.45.1",
                0.016483575000893324,
                0.01987634986980641,
                True,
            ),
        },
    ),
    (
        22,
        178,
        "042ea9c5850a1f1f4a681074b8811163796fb04aeb14327d2a4553db392f502a",
        "bde728e3d94982379a5092b2ad804e3481de1c6c89e7a6f8fc1b8a3ee11c6659",
        {
            "flagquantum_native": (
                "FlagQuantum",
                "0.2.0",
                0.3002971250039991,
                0.034208133706395465,
                True,
            ),
            "qiskit_aer": (
                "Qiskit Aer",
                "0.17.2",
                0.5344154083984904,
                0.02697212014385453,
                True,
            ),
            "cirq_simulator": (
                "Cirq Simulator",
                "1.7.0",
                0.8590032666979823,
                0.021603352070167947,
                True,
            ),
            "pennylane_lightning_qubit": (
                "PennyLane Lightning",
                "0.45.1",
                0.34113924580160526,
                0.02631843158320086,
                True,
            ),
        },
    ),
    (
        24,
        194,
        "1c86cd82d2e89171b1784d9a95ffd22facfee0f76dd31e770506bf87df187a22",
        "2808ec5cd1b9c929c2a54dc1ed06d91694c064b8cd5dbe80a3098b2a7dfeac26",
        {
            "flagquantum_native": (
                "FlagQuantum",
                "0.2.0",
                1.1646358207974117,
                0.03273664257576115,
                True,
            ),
            "qiskit_aer": (
                "Qiskit Aer",
                "0.17.2",
                2.2803735540946946,
                0.021090661837981747,
                True,
            ),
            "cirq_simulator": (
                "Cirq Simulator",
                "1.7.0",
                3.923549395799637,
                0.05235069812666371,
                True,
            ),
            "pennylane_lightning_qubit": (
                "PennyLane Lightning",
                "0.45.1",
                1.6104889291978908,
                0.02778322681139185,
                True,
            ),
        },
    ),
)


def bundled_report() -> dict[str, Any]:
    """Return an isolated copy of the packaged evidence projection."""

    engine_order = (
        "flagquantum_native",
        "qiskit_aer",
        "cirq_simulator",
        "pennylane_lightning_qubit",
    )
    rows = []
    for n_wires, gate_count, fingerprint, ir_content_hash, measurements in _ROWS:
        engines = {}
        native_seconds = measurements["flagquantum_native"][2]
        for engine in engine_order:
            label, version, median, relative_mad, stable = measurements[engine]
            engines[engine] = {
                "label": label,
                "version": version,
                "median_seconds": median,
                "relative_median_absolute_deviation": relative_mad,
                "stable": stable,
                "correctness_passed": True,
            }
        rows.append(
            {
                "workload": {
                    "name": "hardware_efficient_statevector",
                    "n_wires": n_wires,
                    "layers": 2,
                    "gate_count": gate_count,
                    "batch_size": 1,
                    "dtype": "complex128",
                    "seed": 7319,
                },
                "workload_fingerprint": fingerprint,
                "ir_content_hash": ir_content_hash,
                "engines": engines,
                "ratios_to_flagquantum": {
                    engine: measurements[engine][2] / native_seconds
                    for engine in engine_order
                },
            }
        )
    return deepcopy(
        {
            "schema": "flagquantum.simulator_comparison_report.v1",
            "passed": True,
            "correctness_passed": True,
            "non_release_evidence": True,
            "release_gate_allowed": False,
            "reference_engine": "flagquantum_native",
            "engine_order": list(engine_order),
            "comparison_identity": _IDENTITY,
            "rows": rows,
            "sources": [
                {"path": SOURCE_PATH, "sha256": SOURCE_SHA256},
                {
                    "path": WORKLOAD_MANIFEST_PATH,
                    "sha256": WORKLOAD_MANIFEST_SHA256,
                },
            ],
        }
    )


__all__ = (
    "SOURCE_PATH",
    "SOURCE_SHA256",
    "WORKLOAD_MANIFEST_PATH",
    "WORKLOAD_MANIFEST_SHA256",
    "bundled_report",
)
