"""Replaceable SDK and embedded adapters for independent DEM sampling."""

from __future__ import annotations

import ctypes
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import torch

from ...errors import CapabilityError
from . import engine
from .measurement_sampler import _as_torch_sample

_STRIPE_SHOTS = 1024
_MAX_PAYLOAD = 64 * 1024**2


class _Stream(Protocol):
    def sample(
        self,
        shots: int,
        *,
        packed: bool,
        errors: torch.Tensor | None,
        return_errors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    def sample_write(
        self,
        shots: int,
        outputs: tuple[str | None, str | None, str | None],
        formats: tuple[str, str, str],
        replay: str | None,
        replay_format: str,
    ) -> None: ...


def _principal_bytes(
    shots: int,
    widths: tuple[int, int, int],
    packed: bool,
    return_errors: bool,
    replay_packed: bool | None,
) -> int:
    padded = sum(((n + 1023) // 1024) * 1024 for n in widths)
    native = padded * _STRIPE_SHOTS // 8
    d, o, e = widths
    output = shots * sum(
        (n + 7) // 8 if packed else n for n in (d, o, e)[: 3 if return_errors else 2]
    )
    replay = (
        0 if replay_packed is None else shots * ((e + 7) // 8 if replay_packed else e)
    )
    return 4 * native + 3 * output + 3 * replay + 65536


def _check_budget(
    shots: int,
    widths: tuple[int, int, int],
    packed: bool,
    return_errors: bool,
    replay_packed: bool | None,
) -> None:
    if (
        _principal_bytes(shots, widths, packed, return_errors, replay_packed)
        > _MAX_PAYLOAD
    ):
        raise CapabilityError("DEM sampling principal payload exceeds the 64 MiB limit")


class _SdkSampler(Protocol):
    def sample(
        self,
        shots: int,
        *,
        bit_packed: bool,
        return_errors: bool,
        recorded_errors_to_replay: object,
    ) -> object: ...

    def sample_write(
        self,
        shots: int,
        *,
        det_out_file: str | None,
        det_out_format: str,
        obs_out_file: str | None,
        obs_out_format: str,
        err_out_file: str | None,
        err_out_format: str,
        replay_err_in_file: str | None,
        replay_err_in_format: str,
    ) -> None: ...


class _SdkStream:
    def __init__(
        self, text: str, seed: int | None, widths: tuple[int, int, int]
    ) -> None:
        sdk = engine._stim()
        if sdk.__version__ != "1.16.0":
            raise CapabilityError("compiled DEM sampling requires Stim 1.16.0")
        sampler: object = sdk.DetectorErrorModel(text).compile_sampler(seed=seed)
        if not callable(getattr(sampler, "sample", None)) or not callable(
            getattr(sampler, "sample_write", None)
        ):
            raise CapabilityError("invalid Stim DEM sampler")
        self._sampler = cast(_SdkSampler, sampler)
        self._widths = widths

    def sample(
        self,
        shots: int,
        *,
        packed: bool,
        errors: torch.Tensor | None,
        return_errors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raw = self._sampler.sample(
            shots,
            bit_packed=packed,
            return_errors=return_errors,
            recorded_errors_to_replay=None if errors is None else errors.numpy(),
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise CapabilityError("invalid SDK DEM sample result")
        det, obs, err = raw
        return (
            _as_torch_sample(det).clone(),
            _as_torch_sample(obs).clone(),
            None if err is None else _as_torch_sample(err).clone(),
        )

    def sample_write(
        self,
        shots: int,
        outputs: tuple[str | None, str | None, str | None],
        formats: tuple[str, str, str],
        replay: str | None,
        replay_format: str,
    ) -> None:
        self._sampler.sample_write(
            shots,
            det_out_file=outputs[0],
            det_out_format=formats[0],
            obs_out_file=outputs[1],
            obs_out_format=formats[1],
            err_out_file=outputs[2],
            err_out_format=formats[2],
            replay_err_in_file=replay if self._widths[2] else None,
            replay_err_in_format=replay_format,
        )


class _EmbeddedStream:
    def __init__(
        self, text: str, seed: int | None, widths: tuple[int, int, int]
    ) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        self._handle: object = self._native.prepare_dem(text, seed)
        self._widths = widths

    def sample(
        self,
        shots: int,
        *,
        packed: bool,
        errors: torch.Tensor | None,
        return_errors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        replay = (
            None
            if errors is None
            else bytearray(ctypes.string_at(errors.data_ptr(), errors.numel()))
        )
        raw: object = self._native.sample_dem(
            self._handle,
            shots,
            packed,
            replay,
            errors is not None and errors.dtype == torch.uint8,
            return_errors,
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise CapabilityError("invalid embedded DEM sample result")
        result: list[torch.Tensor | None] = []
        for i, (value, width) in enumerate(zip(raw, self._widths, strict=True)):
            columns = (width + 7) // 8 if packed else width
            if i == 2 and not return_errors:
                if value is not None:
                    raise CapabilityError("unexpected embedded error result")
                result.append(None)
            elif not isinstance(value, bytearray) or len(value) != shots * columns:
                raise CapabilityError("invalid embedded DEM output buffer")
            elif columns:
                result.append(
                    torch.frombuffer(
                        value, dtype=torch.uint8 if packed else torch.bool
                    ).reshape(shots, columns)
                )
            else:
                result.append(
                    torch.empty((shots, 0), dtype=torch.uint8 if packed else torch.bool)
                )
        det, obs, err = result
        if det is None or obs is None:
            raise CapabilityError("missing embedded DEM output")
        return det, obs, err

    def sample_write(
        self,
        shots: int,
        outputs: tuple[str | None, str | None, str | None],
        formats: tuple[str, str, str],
        replay: str | None,
        replay_format: str,
    ) -> None:
        self._native.write_dem(
            self._handle,
            shots,
            outputs[0],
            formats[0],
            outputs[1],
            formats[1],
            outputs[2],
            formats[2],
            replay,
            replay_format,
        )


def prepare(text: str, seed: int | None, widths: tuple[int, int, int]) -> _Stream:
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        try:
            return _EmbeddedStream(text, seed, widths)
        except ImportError as error:
            raise CapabilityError(
                "embedded DEM sampler extension is unavailable"
            ) from error
    return _SdkStream(text, seed, widths)
