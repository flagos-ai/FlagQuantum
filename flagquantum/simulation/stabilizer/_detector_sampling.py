"""Bounded native detector execution and exact error analysis of source IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from ...core.ir import CircuitIR
from ...errors import CapabilityError
from .engine import _stim, _stim_experiment_dem_text, _validate_stim_reference
from .measurement_sampler import _as_torch_sample, _lower_program, _require_within_limit

_MAX_ANALYSIS_CELLS = 1000000
_MAX_DEM_BYTES = 16 * 1024**2


class _DetectorStream(Protocol):
    def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def sample_write(
        self,
        shots: int,
        *,
        filepath: str,
        format: str,
        prepend_observables: bool,
        append_observables: bool,
        obs_out_filepath: str | None,
        obs_out_format: str,
    ) -> None: ...


def _source_text(
    ir: CircuitIR,
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
    *,
    analysis: bool = False,
) -> str:
    operations, measurements = _lower_program(ir, allow_pauli_noise=True)
    outputs = len(detector_columns) + len(observable_columns)
    branches = sum(
        4 ** len(g.wires) - 1 for g in ir.instructions if g.metadata.get("is_channel")
    )
    # Sampler preparation also runs noiseless DEM analysis to reject gauge parities.
    _require_within_limit(
        "DEM support cells", (ir.n_wires + measurements) * outputs, _MAX_ANALYSIS_CELLS
    )
    if analysis:
        _require_within_limit(
            "DEM fault support cells", branches * outputs, _MAX_ANALYSIS_CELLS
        )
        # Each admitted fault can name every parity; include line/probability overhead.
        _require_within_limit(
            "DEM text estimate bytes",
            branches * (128 + 24 * outputs) + 64 * outputs,
            _MAX_DEM_BYTES,
        )
    lines = []
    for name, wires, arguments in operations:
        if arguments is not None:
            if not isinstance(arguments, tuple):
                raise CapabilityError("unsupported detector noise arguments")
            name += "(" + ",".join(map(repr, arguments)) + ")"
        targets = " ".join(f"rec[{w}]" if w < 0 else str(w) for w in wires)
        lines.append(f"{name} {targets}")
    for row in detector_columns:
        lines.append("DETECTOR " + " ".join(f"rec[{c - measurements}]" for c in row))
    for index, row in enumerate(observable_columns):
        lines.append(
            f"OBSERVABLE_INCLUDE({index}) "
            + " ".join(f"rec[{c - measurements}]" for c in row)
        )
    return "\n".join(lines)


class _SdkDetectorSampler:
    def __init__(self, text: str, seed: int | None) -> None:
        stim = _stim()
        if stim.__version__ != "1.16.0":
            raise CapabilityError("detector execution requires Stim 1.16.0")
        circuit = stim.Circuit(text)
        _validate_stim_reference(circuit)
        self._sampler: Any = circuit.compile_detector_sampler(seed=seed)

    def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self._sampler.sample(shots, separate_observables=True)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise CapabilityError("detector engine returned an invalid pair")
        return _as_torch_sample(pair[0]).clone(), _as_torch_sample(pair[1]).clone()

    def sample_write(
        self,
        shots: int,
        *,
        filepath: str,
        format: str,
        prepend_observables: bool,
        append_observables: bool,
        obs_out_filepath: str | None,
        obs_out_format: str,
    ) -> None:
        self._sampler.sample_write(
            shots,
            filepath=filepath,
            format=format,
            prepend_observables=prepend_observables,
            append_observables=append_observables,
            obs_out_filepath=obs_out_filepath,
            obs_out_format=obs_out_format,
        )


def _prepare_detector_sampler(
    ir: CircuitIR,
    *,
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
    seed: int | None,
) -> _DetectorStream:
    text = _source_text(ir, detector_columns, observable_columns)
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        from ._embedded_sampler import _EmbeddedDetectorSampler

        try:
            return _EmbeddedDetectorSampler(
                text, seed, len(detector_columns), len(observable_columns)
            )
        except (ValueError, RuntimeError) as error:
            raise CapabilityError(
                "detector preparation failed or has nondeterministic declared parities"
            ) from error
    return _SdkDetectorSampler(text, seed)


def _exact_detector_error_model_text(
    ir: CircuitIR,
    *,
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> str:
    text = _source_text(ir, detector_columns, observable_columns, analysis=True)
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        from importlib import import_module

        native = import_module(f"{__package__}._embedded_stim")
        try:
            result: object = native.detector_error_model(text)
        except (ValueError, RuntimeError) as error:
            raise CapabilityError(
                "circuit has no supported exact DEM conversion"
            ) from error
        if not isinstance(result, str):
            raise CapabilityError("embedded DEM returned an invalid text result")
        # Match the existing SDK helper's shape declarations, without another parser.
        declarations = [f"detector D{i}" for i in range(len(detector_columns))]
        declarations += [
            f"logical_observable L{i}" for i in range(len(observable_columns))
        ]
        result = "\n".join((result, *declarations)).strip()
    else:
        if _stim().__version__ != "1.16.0":
            raise CapabilityError("detector execution requires Stim 1.16.0")
        result = _stim_experiment_dem_text(text, decompose_errors=False)
    _require_within_limit("DEM text bytes", len(result.encode("utf-8")), _MAX_DEM_BYTES)
    return result
