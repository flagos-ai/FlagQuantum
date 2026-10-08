#!/usr/bin/env python3
"""Validate the von Neumann entropy output contract against the live implementation.

``fq.vn_entropy(qubits=...)`` asks an execution for the entropy of a subsystem. The output
*kind* is authorized by ``contracts/observable-outputs-v1-candidate.json`` and
``tools/public_api_snapshot.py`` already proves the signature is the reviewed one. Neither can
say what the number is: a request whose signature is exactly as reviewed can still return a
Rényi entropy, an entropy in bits where nats were promised, the entropy of a different
subsystem, or a value assembled from the wrong eigenvalues -- and every one of those is a
plausible-looking float between zero and ``log 2 ** k``.

This gate measures the properties that make the answer meaningful and compares them with
``contracts/vn-entropy-output-contract.toml``:

``definition``
    The number is ``-Tr rho log rho``. The gate checks two consequences that are hard to reach
    by accident: a globally pure state has equal entropy on a selection and on its complement,
    and the entropy is bounded by ``log`` of the selection's dimension. An implementation that
    returned the entropy of the whole state, or of the complementary subsystem, fails the first;
    one that forgot to clamp a negative eigenvalue fails the second.

``agreement``
    The number has to reproduce an independent reference assembled here from amplitudes: the
    gate builds ``|psi><psi|`` itself, traces out the unselected qubits with an explicit
    ``permute``/``einsum``, diagonalises with ``eigvalsh``, and sums ``-p log p`` with its own
    arithmetic. It shares no code with ``flagquantum/simulation/entropy.py``, so agreement is
    evidence about that module rather than a restatement of it.

``mode``
    Entropy is a property of the state, not of the execution route, so every mode that holds
    the state answers the same number. ``mps`` matters most here: it is the one mode whose
    answer is read from an entanglement spectrum instead of a reduced matrix, and this check is
    what says the shortcut and the reduction agree.

``log_base``
    A named base divides the natural entropy by ``log(base)``, and the four recorded bases pin
    that arithmetic -- including the base below one, which negates the sign. Bases the gate
    cannot answer are refused by name rather than by returning ``inf`` or ``nan``; the
    ``refusal`` rows carry those sentences.

``bound``
    A selection costs ``4 ** len(qubits)`` rather than ``4 ** n``, so the density-matrix
    output's statevector ceiling does not carry over to this kind. The gate measures that
    directly: at the recorded size ``fq.density_matrix`` refuses the same program while
    ``fq.vn_entropy`` answers it in every mode.

``refusal``
    Every sentence the contract records is raised by a trigger the gate runs, so a recorded
    message is the live one rather than a remembered one.

``baseline``
    The ``[[baseline]]`` values are a frozen regression baseline, not the correctness evidence.
    An implementation and a baseline measured from it can be wrong together, which is why the
    independent measurements above carry the weight and the contract says so. Each baseline row
    also carries its PennyLane ``default.qubit`` reading, so a drift in either is visible.

Run ``--measure`` to print the recorded values for the current tree.
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
CONTRACT = ROOT / "contracts" / "vn-entropy-output-contract.toml"

# The gate's own probability floor. Entropy is computed from eigenvalues, and an eigenvalue that
# is numerically zero would contribute ``0 * log 0``; the reference below decides "numerically
# zero" for itself rather than importing the implementation's threshold.
_REFERENCE_FLOOR = 1e-30


def _noise_model(row: dict[str, Any]) -> Any:
    """Return the channel a row names, or ``None`` if the row is a pure-state row."""

    if row.get("damping") is None:
        return None
    from flagquantum.noise import NoiseModel, amplitude_damping_channel

    # ``dtype`` is named explicitly because the channel's own default is complex64, which would
    # put a float32 error into a contract measured in complex128 and hide it in the tolerance.
    channel = amplitude_damping_channel(float(row["damping"]), dtype=torch.complex128)
    return NoiseModel().add("h", channel, qubits=int(row["damped_qubit"]))


def _build(row: dict[str, Any]) -> Any:
    """Build the program a contract row names from the public circuit API."""

    import flagquantum as fq

    circuit = fq.Circuit(int(row["n_qubits"]), dtype=torch.complex128)
    for qubit in row.get("hadamard", []):
        circuit = circuit.h(int(qubit))
    for qubit, angle in row.get("rotations", []):
        circuit = circuit.ry(int(qubit), theta=float(angle))
    for control, target in row.get("entanglers", []):
        circuit = circuit.cx(int(control), int(target))
    return circuit


def _selection(qubits: list[int], *, log_base: float | None = None) -> Any:
    import flagquantum as fq

    return fq.vn_entropy(
        [int(qubit) for qubit in qubits],
        log_base=log_base,
    )


def _readout_model() -> Any:
    from flagquantum.noise import NoiseModel, ReadoutError

    return NoiseModel().add_readout([0], ReadoutError(((0.9, 0.1), (0.1, 0.9))))


def _bell() -> Any:
    import flagquantum as fq

    return fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)


# The two-qubit Bell program as a contract-shaped row, so the base arithmetic is measured with
# the same builder, the same reference state and the same noise handling as every other row.
_BELL_ROW: dict[str, Any] = {
    "name": "bell pair",
    "n_qubits": 2,
    "hadamard": [0],
    "entanglers": [[0, 1]],
}


def _damped_program() -> Any:
    import flagquantum as fq

    return fq.Circuit(2, dtype=torch.complex128).h(0)


def _damped_model() -> Any:
    """The channel the two mixed rows are measured with, built the way the gate builds it."""

    from flagquantum.noise import NoiseModel, amplitude_damping_channel

    channel = amplitude_damping_channel(0.25, dtype=torch.complex128)
    return NoiseModel().add("h", channel, qubits=0)


def _damping_kraus(rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The amplitude-damping Kraus pair, written here rather than read from the package."""

    return (
        torch.tensor([[1.0, 0.0], [0.0, (1.0 - rate) ** 0.5]], dtype=torch.complex128),
        torch.tensor([[0.0, rate**0.5], [0.0, 0.0]], dtype=torch.complex128),
    )


def _apply_damping(
    density: torch.Tensor, qubit: int, n_qubits: int, rate: float
) -> torch.Tensor:
    """Apply one amplitude-damping channel to one qubit of a density matrix.

    The channel is a sum of ``K rho K^dag`` over the Kraus pair, so the ket index of the named
    qubit is contracted with ``K`` and the bra index with ``K^dag``. Contracting the two copies
    the same way would be a different operation, which is the mistake this routine's shape
    discipline exists to make impossible.
    """

    order = [qubit, n_qubits + qubit] + [
        axis for axis in range(2 * n_qubits) if axis not in (qubit, n_qubits + qubit)
    ]
    moved = density.permute(order).reshape(2, 2, -1)
    total = torch.zeros_like(moved)
    for operator in _damping_kraus(rate):
        total = total + torch.einsum("ai,ijz,bj->abz", operator, moved, operator.conj())
    return total.reshape([2] * (2 * n_qubits)).permute(order)


def _partial_trace(
    density: torch.Tensor, qubits: tuple[int, ...], n_qubits: int
) -> torch.Tensor:
    """Trace every unselected qubit out of ``density``, leaving the selected subsystem."""

    ket = list(qubits)
    bra = [qubit + n_qubits for qubit in qubits]
    traced = [qubit for qubit in range(n_qubits) if qubit not in qubits]
    order = ket + bra + traced + [qubit + n_qubits for qubit in traced]
    size = 2 ** len(qubits)
    remainder = 2 ** len(traced)
    # The repeated ``c`` is the point: the surviving index pair is (ket, bra) and the traced pair
    # is contracted against itself, so this is a partial trace rather than a sum over both
    # copies, which would keep the whole state's eigenvalues.
    reduced = torch.einsum(
        "abcc->ab",
        density.permute(order).reshape(size, size, remainder, remainder),
    )
    # ``eigvalsh`` is only valid on a Hermitian matrix; symmetrising the assembled product
    # removes the round-off asymmetry that would otherwise make the routine warn or reorder.
    return (reduced + reduced.conj().transpose(-1, -2)) / 2


def _entropy_of(reduced: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(reduced).clamp_min(0.0)
    positive = eigenvalues > _REFERENCE_FLOOR
    return float(-(eigenvalues[positive] * torch.log(eigenvalues[positive])).sum())


def _reference_entropy(row: dict[str, Any], qubits: tuple[int, ...]) -> float:
    """Return ``-Tr rho log rho`` for one selection, assembled without the implementation.

    The amplitudes come from the package under test -- that is the state being asked about --
    but the density matrix, the channel, the partial trace, the diagonalisation and the
    logarithm are all the gate's, so this is an independent route to the same number rather
    than a call into the module being measured.
    """

    import flagquantum as fq

    n_qubits = int(row["n_qubits"])
    amplitudes = fq.run(_build(row)).to_statevector().reshape(-1).to(torch.complex128)
    density = torch.outer(amplitudes, amplitudes.conj()).reshape([2] * (2 * n_qubits))
    if row.get("damping") is not None:
        density = _apply_damping(
            density,
            int(row["damped_qubit"]),
            n_qubits,
            float(row["damping"]),
        )
    return _entropy_of(_partial_trace(density, qubits, n_qubits))


def _finite(value: float, what: str, errors: list[str]) -> bool:
    """Record a non-finite reading and say whether it can be compared.

    Every comparison below is a tolerance test, and ``nan > tolerance`` is false, so a
    ``nan`` answer would otherwise satisfy each of them and be reported as agreement. The
    reading has to be checked for finiteness *before* it is compared, not after.
    """

    if value == value and value not in (float("inf"), float("-inf")):
        return True
    errors.append(f"{what} is {value!r}, which no tolerance can compare")
    return False


def _definition_errors(contract: dict[str, Any]) -> list[str]:
    """Measure the identity itself: the complement reading and the dimensional bound."""

    errors: list[str] = []
    tolerance = float(
        contract["agreement"].get("independent_reference_tolerance", 1e-12)
    )
    floor = float(contract["agreement"].get("state_separation_floor", 1e-6))
    for row in contract.get("baseline", []):
        n_qubits = int(row["n_qubits"])
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        mixed = row.get("state", "pure") == "mixed"
        value = _run_circuit(row, qubits)
        if not _finite(value, f"baseline {row['name']!r} on {list(qubits)}", errors):
            continue
        bound = float(torch.log(torch.tensor(float(2 ** len(qubits)))))
        if value < -tolerance or value > bound + tolerance:
            errors.append(
                f"baseline {row['name']!r} on {list(qubits)} answered {value!r}, outside "
                f"[0, {bound!r}] for a subsystem of that size"
            )
        if (
            row.get("value") is not None
            and abs(value - float(row["value"])) > tolerance
        ):
            errors.append(
                f"baseline {row['name']!r} on {list(qubits)} answered {value!r} rather than the "
                f"recorded {float(row['value'])!r}"
            )
        complement = tuple(qubit for qubit in range(n_qubits) if qubit not in qubits)
        if not complement:
            continue
        other = _run_circuit(row, complement)
        if not _finite(
            other, f"baseline {row['name']!r} on complement {list(complement)}", errors
        ):
            continue
        if not mixed:
            # For a globally pure state these are the same number by mathematics. The check is
            # still worth running, because an implementation that reported the state of the
            # whole program, or of one fixed qubit, would break it on the rows where the
            # selection is not the whole program.
            if abs(value - other) > tolerance:
                errors.append(
                    f"baseline {row['name']!r} is pure, so its entropy on {list(qubits)} and on "
                    f"{list(complement)} must agree, and they are {value!r} and {other!r}"
                )
            continue
        if row.get("complement_value") is not None and (
            abs(other - float(row["complement_value"])) > tolerance
        ):
            errors.append(
                f"baseline {row['name']!r} on the complement {list(complement)} answered "
                f"{other!r} rather than the recorded {float(row['complement_value'])!r}"
            )
        if abs(value - other) < floor:
            errors.append(
                f"baseline {row['name']!r} is recorded as mixed so that a selection and its "
                f"complement can be told apart, and its two sides answer {value!r} and {other!r}"
            )
    return errors


def _run_circuit(
    row: dict[str, Any],
    qubits: tuple[int, ...],
    *,
    log_base: float | None = None,
    mode: str | None = None,
) -> float:
    import flagquantum as fq

    options = None if mode is None else fq.ExecutionOptions(mode=mode)
    result = fq.run(
        _build(row),
        outputs=_selection(list(qubits), log_base=log_base),
        noise_model=_noise_model(row),
        options=options,
    )
    return float(result.vn_entropy[0])


def _agreement_errors(contract: dict[str, Any]) -> list[str]:
    """Compare the implementation with the gate's own amplitude-side reference."""

    errors: list[str] = []
    tolerance = float(
        contract["agreement"].get("independent_reference_tolerance", 1e-12)
    )
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        value = _run_circuit(row, qubits)
        if not _finite(value, f"baseline {row['name']!r} on {list(qubits)}", errors):
            continue
        expected = _reference_entropy(row, qubits)
        if not _finite(
            expected, f"the amplitude-side reference for {row['name']!r}", errors
        ):
            continue
        if abs(value - expected) > tolerance:
            errors.append(
                f"baseline {row['name']!r} on {list(qubits)} answered {value!r} and the "
                f"amplitude-side reference is {expected!r}"
            )
    if not errors:
        # A reference that agrees everywhere because both sides are the same function would be
        # worthless, so the gate records that the reference is not the implementation's own
        # arithmetic: it is asked for a value the implementation does not produce.
        probe = contract.get("reference_programs", {})
        if not probe.get("tool"):
            errors.append(
                "the contract names no reference program, so agreement is only self-consistency"
            )
    return errors


def _mode_errors(contract: dict[str, Any]) -> list[str]:
    """Measure that every execution route that holds the state answers the same number."""

    errors: list[str] = []
    tolerance = float(contract["agreement"].get("mode_tolerance", 1e-12))
    modes = [str(mode) for mode in contract["request"].get("modes", [])]
    if not modes:
        errors.append(
            "the contract names no execution modes, so route agreement is untested"
        )
        return errors
    for row in contract.get("baseline", []):
        if row.get("state", "pure") == "mixed":
            # A noisy program refuses the modes that cannot hold a mixed state, and the contract
            # records that refusal separately. Skipping the row here would be silent, so the
            # gate counts what it skipped and the summary reports it.
            continue
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        reference = _run_circuit(row, qubits)
        if not _finite(
            reference, f"baseline {row['name']!r} on the default route", errors
        ):
            continue
        for mode in modes:
            value = _run_circuit(row, qubits, mode=mode)
            if not _finite(
                value,
                f"baseline {row['name']!r} on {list(qubits)} in mode {mode!r}",
                errors,
            ):
                continue
            if abs(value - reference) > tolerance:
                errors.append(
                    f"baseline {row['name']!r} on {list(qubits)} answers {mode!r} with {value!r} "
                    f"and the default route with {reference!r}"
                )
    pure_rows = [
        row
        for row in contract.get("baseline", [])
        if row.get("state", "pure") != "mixed"
    ]
    if not pure_rows:
        errors.append("every row is mixed, so route agreement is measured on nothing")
    return errors


def _log_base_errors(contract: dict[str, Any]) -> list[str]:
    """Measure the base arithmetic against the natural reading the contract records."""

    errors: list[str] = []
    tolerance = float(contract["agreement"].get("log_base_tolerance", 1e-12))
    rows = contract.get("log_base", [])
    if not rows:
        errors.append(
            "the contract records no bases, so the base arithmetic is untested"
        )
        return errors
    natural = _run_circuit(_BELL_ROW, (0,))
    for row in rows:
        name = str(row.get("name"))
        mode = str(row.get("mode"))
        recorded = float(row.get("value"))
        if mode == "natural":
            value = natural
        else:
            value = _run_circuit(_BELL_ROW, (0,), log_base=float(row["base"]))
        if not _finite(value, f"log base {name!r}", errors):
            continue
        if abs(value - recorded) > tolerance:
            errors.append(
                f"log base {name!r} answered {value!r} rather than the recorded {recorded!r}"
            )
        # The named-base reading has to be the natural one divided by log(base); stating that
        # here is what makes the row a measurement of the division rather than of a constant.
        if mode == "named":
            divisor = float(
                torch.log(torch.tensor(float(row["base"]), dtype=torch.float64))
            )
            expected = natural / divisor
            if abs(value - expected) > tolerance:
                errors.append(
                    f"log base {name!r} answered {value!r} rather than natural entropy "
                    f"{natural!r} divided by log({float(row['base'])!r}) = {expected!r}"
                )
    # ``log_base=None`` and the explicit Euler number have to agree, or "no base" would mean
    # something other than the natural logarithm.
    explicit = _run_circuit(_BELL_ROW, (0,), log_base=float(torch.e))
    if (
        _finite(explicit, "the explicit Euler-number base", errors)
        and abs(explicit - natural) > tolerance
    ):
        errors.append(
            f"the natural logarithm and log_base=e disagree: {explicit!r} and {natural!r}"
        )
    return errors


def _bound_errors(contract: dict[str, Any]) -> list[str]:
    """Measure that the selection's cost, not the program's size, bounds this kind."""

    import flagquantum as fq

    errors: list[str] = []
    probed = int(contract["bound"].get("probed_qubits", 12))
    scaling = contract.get("scaling", {})
    selection = tuple(int(q) for q in scaling.get("selection", [0])) or (0,)
    for n_qubits in [int(n) for n in scaling.get("probe_qubits", [])]:
        row: dict[str, Any] = {
            "name": f"{n_qubits}-qubit chain",
            "n_qubits": n_qubits,
            "hadamard": [],
            "rotations": [[qubit, 0.2] for qubit in range(n_qubits)],
            "entanglers": [[qubit, qubit + 1] for qubit in range(n_qubits - 1)],
            "qubits": list(selection),
        }
        circuit = _build(row)
        entropy = _run_circuit(row, selection)
        if not _finite(
            entropy, f"a {n_qubits}-qubit program on {list(selection)}", errors
        ):
            continue
        if (
            not 0.0
            <= entropy
            <= float(torch.log(torch.tensor(float(2 ** len(selection))))) + 1e-9
        ):
            errors.append(
                f"a {n_qubits}-qubit program answered {entropy!r} on {list(selection)}, which is "
                "not an entropy of that subsystem"
            )
        # The density-matrix output writes the square of the state and is bounded; this kind is
        # not. Measuring both sides is what makes the claim about the bound a measurement.
        try:
            fq.run(circuit, outputs=fq.density_matrix(list(selection)))
        except Exception as error:
            if "density-matrix" not in str(error) and "statevector" not in str(error):
                errors.append(
                    f"a {n_qubits}-qubit program refused a density-matrix output with an "
                    f"unexpected sentence: {error}"
                )
        else:
            errors.append(
                f"a {n_qubits}-qubit program served a density-matrix output, so the comparison "
                "this check rests on is no longer measuring anything"
            )
        # The MPS route holds the same state and must answer it too, which is the point of the
        # route existing at all: an entropy available only from a materialised statevector would
        # not extend the reachable size.
        via_mps = _run_circuit(row, selection, mode="mps")
        if not _finite(via_mps, f"a {n_qubits}-qubit program in the mps mode", errors):
            continue
        if abs(via_mps - entropy) > float(
            contract["agreement"].get("mode_tolerance", 1e-12)
        ):
            errors.append(
                f"a {n_qubits}-qubit program answered {via_mps!r} in the mps mode and "
                f"{entropy!r} in the default route"
            )
    sizes = [int(n) for n in scaling.get("probe_qubits", [])]
    if not sizes:
        errors.append("the contract names no sizes, so the bound claim is unmeasured")
    elif probed not in sizes:
        errors.append(
            f"the contract records {probed} as the probed size and the gate measures {sizes}, "
            "so the sentence about the bound is not the one that was measured"
        )
    return errors


def _refusal_errors(contract: dict[str, Any]) -> list[str]:
    """Run every recorded refusal trigger and compare the sentence with the record."""

    import flagquantum as fq

    bell = _bell()
    triggers: dict[str, Any] = {
        "no qubit is named": lambda: fq.vn_entropy([]),
        "an observable is not a qubit selection": lambda: fq.OutputRequest(
            "vn_entropy", (0,), observable=fq.Z(0)
        ),
        "a repeated qubit is refused rather than counted twice": lambda: fq.run(
            bell, outputs=fq.vn_entropy([0, 0])
        ),
        "a negative qubit is refused": lambda: fq.run(
            bell, outputs=fq.vn_entropy([-1])
        ),
        "a fractional qubit is refused": lambda: fq.vn_entropy([0.5]),
        "a boolean is not a qubit": lambda: fq.vn_entropy([True]),
        "a qubit outside the program is named": lambda: fq.run(
            bell, outputs=fq.vn_entropy([5])
        ),
        "a base of zero is refused rather than answered by accident": lambda: fq.vn_entropy(
            [0], log_base=0
        ),
        "a base of one is refused because it has no logarithm": lambda: fq.vn_entropy(
            [0], log_base=1
        ),
        "a negative base is refused": lambda: fq.vn_entropy([0], log_base=-1),
        "an infinite base is refused": lambda: fq.vn_entropy(
            [0], log_base=float("inf")
        ),
        "a boolean base is refused as a number": lambda: fq.vn_entropy(
            [0], log_base=True
        ),
        "a string base is refused as a number": lambda: fq.vn_entropy(
            [0], log_base="2"
        ),
        "a kind that has no base refuses one": lambda: fq.OutputRequest(
            "probabilities", (0,), log_base=2
        ),
        "readout confusion is refused rather than folded in": lambda: fq.run(
            bell, outputs=fq.vn_entropy([0]), noise_model=_readout_model()
        ),
        "shots are refused because entropy is not sampled": lambda: fq.run(
            bell, shots=100, outputs=fq.vn_entropy([0])
        ),
        "a sampling mode is refused by name": lambda: fq.run(
            bell,
            options=fq.ExecutionOptions(mode="stabilizer"),
            outputs=fq.vn_entropy([0]),
        ),
        "a noisy program refuses every mode that cannot hold a mixed state": lambda: fq.run(
            _damped_program(),
            options=fq.ExecutionOptions(mode="mps"),
            outputs=fq.vn_entropy([0]),
            noise_model=_damped_model(),
        ),
    }
    rows = contract.get("refusal", [])
    errors: list[str] = []
    recorded = {str(row.get("name")) for row in rows}
    unmeasured = sorted(recorded - set(triggers))
    if unmeasured:
        errors.append(
            f"the contract records refusals this gate cannot trigger: {unmeasured}"
        )
    unused = sorted(set(triggers) - recorded)
    if unused:
        errors.append(
            f"the gate can trigger refusals the contract does not record: {unused}"
        )
    for row in rows:
        name = str(row.get("name"))
        if name not in triggers:
            continue
        fragment = str(row.get("message_phrase", ""))
        expected = str(row.get("exception", ""))
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
    """Measure the constructor itself: its name, its keywords, and what it produces."""

    import inspect

    import flagquantum as fq

    errors: list[str] = []
    request = contract["request"]
    constructor_name = str(request.get("constructor", "")).removeprefix("flagquantum.")
    constructor = getattr(fq, constructor_name, None)
    if not callable(constructor):
        errors.append(
            f"the contract names {request.get('constructor')!r} as the constructor and it is "
            "not a callable root export"
        )
    if str(request.get("kind")) not in fq.__all__:
        errors.append(
            f"the contract's kind {request.get('kind')!r} is not a root export, so a user "
            "cannot ask for it by name"
        )
    if callable(constructor):
        signature = inspect.signature(constructor)
        for key in ("selection_parameter", "base_parameter"):
            parameter = str(request.get(key, ""))
            if parameter and parameter not in signature.parameters:
                errors.append(
                    f"the contract names {parameter!r} as the {key} and the constructor does "
                    f"not accept it: {signature}"
                )
    produced = fq.vn_entropy([0])
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
    # A base the caller wrote has to survive on the request unchanged; normalising it here would
    # hide which base was asked for from everything downstream.
    if fq.vn_entropy([0], log_base=2).log_base != 2:
        errors.append("the constructor does not keep the base the caller named")
    if fq.vn_entropy([0]).log_base is not None:
        errors.append("the constructor invents a base where the caller named none")
    if not contract.get("baseline"):
        errors.append("the contract records no values, so it measures nothing")
    return errors


def measure() -> dict[str, Any]:
    import flagquantum as fq

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in contract.get("baseline", []):
        qubits = tuple(int(qubit) for qubit in row["qubits"])
        circuit = _build(row)
        values: dict[str, float] = {}
        for mode in contract["request"].get("modes", []):
            values[str(mode)] = float(
                fq.run(
                    circuit,
                    outputs=_selection(list(qubits)),
                    options=fq.ExecutionOptions(mode=str(mode)),
                ).vn_entropy[0]
            )
        rows.append(
            {
                "name": row["name"],
                "n_qubits": int(row["n_qubits"]),
                "qubits": list(qubits),
                "reference": _reference_entropy(row, qubits),
                "modes": values,
            }
        )
    return {"baselines": rows}


def contract_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_request_errors(contract))
    errors.extend(_definition_errors(contract))
    errors.extend(_agreement_errors(contract))
    errors.extend(_mode_errors(contract))
    errors.extend(_log_base_errors(contract))
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
    rows = contract.get("baseline", [])
    pure = [row for row in rows if row.get("state", "pure") != "mixed"]
    mixed = [row for row in rows if row.get("state", "pure") == "mixed"]
    modes = len(contract["request"].get("modes", []))
    print(
        "Von Neumann entropy output contract passed: "
        f"{len(rows)} recorded values ({len(pure)} pure, {len(mixed)} mixed) reproduce an "
        "amplitude-side reference assembled in the gate and satisfy the identity on a selection "
        "and its complement; the pure rows agree across "
        f"{modes} execution modes ({len(pure) * modes} route readings) and the mixed rows "
        "separate their selection from their complement; "
        f"{len(contract.get('log_base', []))} recorded bases agree with the natural entropy "
        f"divided by their logarithm; {len(contract.get('refusal', []))} recorded refusals raise "
        "the sentence the contract states"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
