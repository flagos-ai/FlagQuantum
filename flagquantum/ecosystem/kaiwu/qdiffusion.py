"""QDiffusion factory binding for the FlagQuantum Kaiwu sampler."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from .sampler import KaiwuSampler

_BuilderResult = TypeVar("_BuilderResult")


def bind_qdiffusion_builder(
    builder: Callable[..., _BuilderResult],
    sampler: KaiwuSampler,
) -> Callable[..., _BuilderResult]:
    """Bind every model built by a plugin factory to one Kaiwu sampler.

    The pinned protein workflow calls its imported ``build_qdiffusion`` factory
    for validation, training, baseline, and guided generation. Replacing that
    callable with this wrapper makes all four paths share the bounded sampler.
    The wrapper discards the workflow's ``sa``/``cim`` construction hints and
    verifies the returned model retained the exact injected sampler object.
    """

    @wraps(builder)
    def bound(*args: Any, **kwargs: Any) -> _BuilderResult:
        supplied = kwargs.get("bm_sampler")
        if supplied is not None and supplied is not sampler:
            raise ValueError("QDiffusion builder already received another BM sampler")
        kwargs["bm_sampler"] = sampler
        kwargs["bm_sampler_type"] = "flagquantum-kaiwu"
        kwargs["bm_sampler_kwargs"] = {}
        result = builder(*args, **kwargs)
        energy_model = getattr(result, "energy_model", None)
        if getattr(energy_model, "sampler", None) is not sampler:
            raise RuntimeError(
                "QDiffusion builder did not retain the injected KaiwuSampler"
            )
        return result

    return bound
