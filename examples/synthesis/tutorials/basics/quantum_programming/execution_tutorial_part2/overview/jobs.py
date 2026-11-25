"""Execute and retrieve actual local sample and expectation records."""

import hashlib
import json
import math
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import flagquantum as fq

from .workflow import SHOTS, observe, sample, sample_projector

Kind = Literal["sample", "observe"]


@dataclass(frozen=True)
class Result:
    kind: Kind
    angle: float
    counts: dict[str, int] | None
    expectation: float | None
    exact_expectation: float | None


def _encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def _save(directory: Path, result: Result) -> str:
    record = _encoded(asdict(result))
    key = hashlib.sha256(record).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.json"
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(record)
            handle.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return key


def lookup(directory: Path, key: str) -> Result:
    if re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("a job ID must be a SHA-256 digest")
    raw = (directory / f"{key}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != key:
        raise ValueError("stored result content differs from the job ID")
    data = json.loads(raw)
    return Result(
        data["kind"],
        data["angle"],
        data["counts"],
        data["expectation"],
        data["exact_expectation"],
    )


def submit_both(circuit: fq.Circuit, angle: float, directory: Path) -> tuple[str, str]:
    """Run and persist both original job kinds on one local executor."""
    if not math.isfinite(angle):
        raise ValueError("the job angle must be finite")
    with ThreadPoolExecutor(max_workers=2) as workers:
        sample_job = workers.submit(sample, circuit, angle, seed=207)
        observe_job = workers.submit(sample_projector, circuit, angle, seed=208)
        counts = sample_job.result()
        value, measured = observe_job.result()
    exact, _ = observe(circuit, angle)
    assert sum(counts.values()) == measured == SHOTS
    return (
        _save(directory, Result("sample", angle, counts, None, None)),
        _save(directory, Result("observe", angle, None, value, exact)),
    )
