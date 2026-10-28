"""Shared fixtures for the QEC suites."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
import torch


@pytest.fixture
def tf32_disabled() -> Iterator[None]:
    """Force TF32 off for the test and restore it by assignment.

    `monkeypatch.setattr` cannot undo this flag on pytest 9.2. `allow_tf32` on
    `torch.backends.cuda.matmul` and `torch.backends.cudnn` resolves through the
    module object rather than an instance `__dict__` entry, and pytest 9.2 copies
    an instance entry when one exists and otherwise records "no previous value"
    and restores by deleting the attribute -- which only the class provides, so
    the deletion raises `AttributeError` at teardown. Assigning the previous
    value back is what these flags support, so that is what this does.
    """

    with contextlib.ExitStack() as stack:
        for module in (torch.backends.cuda.matmul, torch.backends.cudnn):
            previous = module.allow_tf32
            module.allow_tf32 = False
            stack.callback(setattr, module, "allow_tf32", previous)
        yield
