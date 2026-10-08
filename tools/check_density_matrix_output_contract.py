#!/usr/bin/env python3
"""Validate the density-matrix output contract against the live implementation.

``fq.density_matrix(qubits=...)`` asks an execution for the state on those qubits as a matrix.
The output *kind* is authorized by ``contracts/observable-outputs-v1-candidate.json``, and
``tools/public_api_snapshot.py`` already proves the signature is the reviewed one. Neither of
those can say what the matrix is, because a matrix is a numerical object: a request whose
signature is exactly as reviewed can still return a matrix that is a marginal instead of a
trace, in ascending order instead of the order the caller named, or at a cost nobody agreed to.

This gate measures the properties that make the answer meaningful and compares them with
``contracts/density-matrix-output-contract.toml``:

``trace``
    The matrix is the state on the named qubits, so its trace is one. This is what tells a trace
    from a marginal: summing a row or column index rather than contracting the row index against
    the column index returns a matrix whose trace is the whole state's purity, which is one for a
    pure state and therefore passes a Bell-pair check by luck. The gate refuses to proceed if its
    mixed probe turns out to be pure, so the check cannot report agreement it did not measure.

``agreement``
    The reduction has to reproduce what the observable path reports for an operator on the same
    qubits, ``Tr(O rho_S) == <O>``. The reduced matrix is assembled in Simulation from
    amplitudes; the expectation is assembled in Runtime through the observable's own route, and
    the operator is lifted into the full space by explicit Kronecker products here rather than by
    the helper the reduction uses. Agreement is therefore evidence about both, and disagreement
    localises the defect.

``permutation``
    Naming the qubits in another order is one basis permutation of the same matrix. The gate
    builds that permutation from the naming and applies it, so an implementation that returned
    the surviving qubits in ascending order fails here instead of passing by coincidence.

``bound``
    Building a matrix from amplitudes writes ``4 ** n`` entries, the square of the state it is
    built from. The statevector route is bounded and its message names the cheaper route; the
    density-matrix route has already paid for the matrix and is not bounded.

``refusal``
    Every sentence the contract records is raised by a trigger the gate runs, so a recorded
    message is the live one rather than a remembered one.

``baseline``
    The ``[[baseline]]`` matrices are a frozen regression baseline, not the correctness evidence.
    An implementation and a baseline measured from it can be wrong together, which is why the
    independent measurements above carry the weight and the contract says so.

Run ``--measure`` to print the recorded matrices for the current tree.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "density-matrix-output-contract.toml"

# The Pauli matrices as constants, so the reference side of the agreement check does not read
# them from the package that is under test. The observables are still built from ``fq.X`` and
# ``fq.Z``, because what is being compared is the package's own expectation route.
_PAULI: dict[str, torch.Tensor] = {
    "x": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128),
    "y": torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128),
    "z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128),
}


def _observable_for(axis: str) -> Any:
    """Return the ``fq`` observable constructor for one Pauli axis."""

    import flagquantum as fq

    mapping = {"x": fq.X, "y": fq.Y, "z": fq.Z}
    if axis not in mapping:
        raise SystemExit(f"the contract names an axis this gate cannot build: {axis!r}")
    return mapping[axis]


def _build(row: dict[str, Any]) -> Any:
    """Build the program a contract row names from the public circuit API."""

    import flagquantum as fq

    circuit = fq.Circuit(int(row["n_qubits"]), dtype=torch.complex128)
    if row.get("style") == "bell":
        circuit = circuit.h(0).cx(0, 1)
    else:
        for qubit, angle in row.get("rotations", []):
            circuit = circuit.ry(int(qubit), theta=float(angle))
    for control, target in row.get("entanglers", []):
        circuit = circuit.cx(int(control), int(target))
    return circuit


def _selection(qubits: list[int]) -> Any:
    import flagquantum as fq

    return fq.density_matrix(qubits=tuple(int(qubit) for qubit in qubits))


def _matrix(row: dict[str, Any]) -> torch.Tensor:
    real = torch.tensor(row["real"], dtype=torch.float64)
    imag = torch.tensor(row["imag"], dtype=torch.float64)
    return (real + 1j * imag).to(torch.complex128)


def _readout_model() -> Any:
    from flagquantum.noise import NoiseModel, ReadoutError

    return NoiseModel().add_readout([0], ReadoutError(((0.9, 0.1), (0.1, 0.9))))


def measure() -> dict[str, Any]:
    import flagquantum as fq

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        value = fq.run(_build(row), outputs=_selection(list(qubits))).density_matrix[0]
        rows.append(
            {
                "name": row["name"],
                "n_qubits": int(row["n_qubits"]),
                "qubits": list(qubits),
                "real": [[float(entry.real) for entry in line] for line in value],
                "imag": [[float(entry.imag) for entry in line] for line in value],
            }
        )
    return {"baselines": rows}


def _trace_errors(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    reduction = contract["reduction"]
    target = float(reduction.get("trace_value", 1.0))
    tolerance = float(reduction.get("trace_tolerance", 1e-10))

    # The probe has to be mixed, or a marginal would pass this check as easily as a trace.
    mixed = (
        fq.Circuit(2, dtype=torch.complex128).ry(0, theta=0.7).ry(1, theta=1.1).cx(0, 1)
    )
    reduced = fq.run(mixed, outputs=_selection([0])).density_matrix[0]
    purity = complex(torch.einsum("ij,ji->", reduced, reduced))
    if abs(purity.real - 1.0) < 1e-6:
        errors.append(
            "the mixed probe is pure, so it cannot tell a trace from a marginal and the check "
            "would report agreement it did not measure"
        )

    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        value = fq.run(_build(row), outputs=_selection(list(qubits))).density_matrix[0]
        trace = complex(torch.trace(value))
        if abs(trace - target) > tolerance:
            errors.append(
                f"baseline {row['name']!r} on qubits {list(qubits)} has trace {trace!r} rather "
                f"than {target}"
            )
        expected = _matrix(row)
        if tuple(value.shape) != tuple(expected.shape):
            errors.append(
                f"baseline {row['name']!r} answers with shape {tuple(value.shape)} rather than "
                f"{tuple(expected.shape)}"
            )
            continue
        difference = float(torch.max(torch.abs(value - expected)))
        if difference > tolerance:
            errors.append(
                f"baseline {row['name']!r} differs from the recorded matrix by {difference:.3e}"
            )
    return errors


def _permutation_matrix(n_qubits: int, order: tuple[int, ...]) -> torch.Tensor:
    """Return the basis permutation that reorders ``sorted(order)`` into ``order``."""

    surviving = sorted(order)
    dimension = 2 ** len(surviving)
    permutation = torch.zeros((dimension, dimension), dtype=torch.complex128)
    for index in range(dimension):
        position = 0
        for qubit in order:
            bit = (index >> (len(order) - 1 - surviving.index(qubit))) & 1
            position = (position << 1) | bit
        permutation[index, position] = 1.0
    return permutation


def _permutation_errors(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    tolerance = float(contract["reduction"].get("permutation_tolerance", 1e-10))
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        if len(qubits) < 2 or tuple(sorted(qubits)) == qubits:
            continue
        ascending = fq.run(
            _build(row), outputs=_selection(sorted(qubits))
        ).density_matrix[0]
        recorded = fq.run(_build(row), outputs=_selection(list(qubits))).density_matrix[
            0
        ]
        permutation = _permutation_matrix(int(row["n_qubits"]), qubits)
        expected = permutation @ ascending @ permutation.T
        difference = float(torch.max(torch.abs(recorded - expected)))
        if difference > tolerance:
            errors.append(
                f"baseline {row['name']!r} answers {list(qubits)} with a matrix that is not the "
                f"basis permutation of {sorted(qubits)}: it differs by {difference:.3e}"
            )
    return errors


def _agreement_errors(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    tolerance = float(contract["reduction"].get("operator_tolerance", 1e-10))
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        circuit = _build(row)
        reduced = fq.run(circuit, outputs=_selection(list(qubits))).density_matrix[0]
        for axes in (("z",) * len(qubits), ("x",) + ("z",) * (len(qubits) - 1)):
            observable = _observable_for(axes[0])(qubits[0])
            for qubit, axis in zip(qubits[1:], axes[1:], strict=True):
                observable = observable @ _observable_for(axis)(qubit)
            reported = complex(
                fq.run(circuit, outputs=fq.expectation(observable)).expectations[0]
            )
            local = _PAULI[axes[0]]
            for axis in axes[1:]:
                local = torch.kron(local, _PAULI[axis])
            from_reduction = complex(torch.einsum("ij,ji->", local, reduced))
            difference = abs(reported - from_reduction)
            if difference > tolerance:
                errors.append(
                    f"baseline {row['name']!r} on qubits {list(qubits)} with axes {axes}: the "
                    f"matrix gives Tr(O rho) = {from_reduction!r} and the observable path "
                    f"reports {reported!r}, a difference of {difference:.3e}"
                )
    return errors


def _bound_errors(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    errors: list[str] = []
    bound = contract["bound"]
    maximum = int(bound.get("max_statevector_qubits", 10))
    probed = int(bound.get("probed_qubits", maximum + 2))
    required = str(bound.get("required_route_in_message", ""))
    if probed <= maximum:
        errors.append(
            "the contract probes a program inside the bound, so the bound is untested"
        )
        return errors
    large = fq.Circuit(probed, dtype=torch.complex128).x(0)
    try:
        fq.run(large, outputs=_selection([0]))
    except ValueError as error:
        if required and required not in str(error):
            errors.append(f"the bound refusal does not name {required!r}: {error}")
    else:
        errors.append(
            f"a {probed}-qubit statevector route built a matrix rather than refusing"
        )
    return errors


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    import flagquantum as fq

    rows = contract.get("refusal", [])
    probed = int(contract["bound"].get("probed_qubits", 12))
    triggers: dict[str, Any] = {
        "an observable is not a qubit selection": lambda: fq.density_matrix(fq.Z(0)),
        "a kind-level observable is refused as well": lambda: fq.OutputRequest(
            "density_matrix", observable=fq.Z(0)
        ),
        "a qubit outside the program is named": lambda: fq.run(
            fq.Circuit(2).h(0), outputs=_selection([5])
        ),
        "a repeated qubit is refused rather than summed": lambda: fq.run(
            fq.Circuit(2).h(0), outputs=_selection([0, 0])
        ),
        "readout confusion is refused rather than folded in": lambda: fq.run(
            _build({"n_qubits": 2, "style": "bell"}),
            outputs=_selection([0]),
            noise_model=_readout_model(),
        ),
        "the statevector route names the smaller route": lambda: fq.run(
            fq.Circuit(probed, dtype=torch.complex128).x(0), outputs=_selection([0])
        ),
    }
    errors: list[str] = []
    recorded = {str(row.get("name")) for row in rows}
    unmeasured = sorted(recorded - set(triggers))
    if unmeasured:
        errors.append(
            f"the contract records refusals this gate cannot trigger: {unmeasured}"
        )
    for row in rows:
        name = str(row.get("name"))
        if name not in triggers:
            continue
        fragment = str(row.get("message_fragment", ""))
        expected = str(row.get("error", ""))
        try:
            triggers[name]()
        except Exception as error:
            if type(error).__name__ != expected:
                errors.append(
                    f"refusal {name!r} raises {type(error).__name__} rather than {expected}"
                )
            elif fragment and fragment not in str(error):
                errors.append(f"refusal {name!r} does not say {fragment!r}: {error}")
        else:
            errors.append(f"refusal {name!r} did not raise")
    return errors


def _request_errors(contract: dict[str, Any]) -> list[str]:
    """Measure the constructor itself: its name, its selection keyword, and its order."""

    import inspect

    import flagquantum as fq

    errors: list[str] = []
    request = contract["request"]
    constructor_name = str(request.get("constructor", "")).removeprefix("flagquantum.")
    constructor = getattr(fq, constructor_name, None)
    if not callable(constructor):
        errors.append(
            f"the contract names {request.get('constructor')!r} as the constructor and it is not "
            "a callable root export"
        )
    if str(request.get("kind")) not in fq.__all__:
        errors.append(
            f"the contract's kind {request.get('kind')!r} is not a root export, so a user cannot "
            "ask for it by name"
        )
    # The selection keyword is the contract's claim about how a user names qubits; a constructor
    # that only accepted a positional argument would satisfy every numerical measurement below.
    parameter = str(request.get("selection_parameter", ""))
    if callable(constructor) and parameter:
        signature = inspect.signature(constructor)
        if parameter not in signature.parameters:
            errors.append(
                f"the contract names {parameter!r} as the selection parameter and the constructor "
                f"does not accept it: {signature}"
            )
    produced = _selection([0])
    if getattr(produced, "kind", None) != request.get("kind"):
        errors.append(
            f"the constructor produces kind {getattr(produced, 'kind', None)!r} rather than "
            f"{request.get('kind')!r}"
        )
    if getattr(produced, "qubits", None) != (0,):
        errors.append(
            f"the constructor records qubits {getattr(produced, 'qubits', None)!r} rather than "
            "the selection the caller named"
        )
    if not contract.get("baseline"):
        errors.append("the contract records no matrices, so it measures nothing")
    return errors


def _route_errors(contract: dict[str, Any]) -> list[str]:
    """Measure that the reduction is a property of the state rather than of the execution route.

    A reduction is arithmetic on amplitudes, so every mode that holds the state has to answer
    with the same matrix. ``auto`` is included because it selects one of the others and must not
    disagree with whichever one it selected.
    """

    import flagquantum as fq

    errors: list[str] = []
    tolerance = float(contract["reduction"].get("trace_tolerance", 1e-10))
    modes = [str(mode) for mode in contract["request"].get("modes", [])]
    if not modes:
        errors.append(
            "the contract names no execution modes, so route agreement is untested"
        )
        return errors
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        circuit = _build(row)
        reference = fq.run(circuit, outputs=_selection(list(qubits))).density_matrix
        for mode in modes:
            value = fq.run(
                circuit,
                outputs=_selection(list(qubits)),
                options=fq.ExecutionOptions(mode=mode),
            ).density_matrix
            difference = float(torch.max(torch.abs(value - reference)))
            if difference > tolerance:
                errors.append(
                    f"baseline {row['name']!r} on qubits {list(qubits)} answers {mode!r} with a "
                    f"matrix that differs from the default route by {difference:.3e}"
                )
    return errors


def contract_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_request_errors(contract))
    errors.extend(_trace_errors(contract))
    errors.extend(_permutation_errors(contract))
    errors.extend(_agreement_errors(contract))
    errors.extend(_route_errors(contract))
    errors.extend(_bound_errors(contract))
    errors.extend(_refusal_errors(contract))
    return errors


def main(argv: list[str]) -> int:
    if "--measure" in argv:
        import json

        print(json.dumps(measure(), indent=2))
        return 0
    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    errors = contract_errors(contract)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Density-matrix output contract passed: "
        f"{len(contract.get('baseline', []))} recorded matrices trace to one, agree with the "
        "observable path, reproduce under the caller's qubit order as a basis permutation and "
        f"across {len(contract['request'].get('modes', []))} execution modes; "
        f"{len(contract.get('refusal', []))} recorded refusals raise the sentence the contract "
        "states"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
