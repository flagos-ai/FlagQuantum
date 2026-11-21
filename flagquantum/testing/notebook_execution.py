# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Shared execution adapted from Qualtran v0.7.0 notebook_execution.py and
# cirq_interop/jupyter_tools.py.

"""Execute documentation notebooks through the existing optional Jupyter tools."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from nbformat import NotebookNode


def execute_notebook_node(
    notebook: NotebookNode,
    *,
    timeout: int = 600,
    kernel_name: str = "python3",
    record_timing: bool = True,
    startup_lock: Path | None = None,
) -> tuple[NotebookNode, dict[str, object]]:
    """Execute a documentation/test notebook and return its conversion resources.

    Jupyter remains the sole execution implementation. An optional file lock
    covers the actual kernel/client startup inside preprocess. The original
    setup context retains ownership of shutdown on success and failure. This
    helper never selects a quantum backend or executes a production plan.
    """
    options: dict[str, object] = {"timeout": timeout, "kernel_name": kernel_name}
    if not record_timing:
        options["record_timing"] = False
    executor = import_module("nbconvert.preprocessors").ExecutePreprocessor(**options)
    if startup_lock is not None:
        setup_kernel = executor.setup_kernel
        lock_type = import_module("filelock").FileLock

        @contextmanager
        def locked_setup(**kwargs: Any) -> Iterator[None]:
            # Enter under the startup lock, execute outside it, then delegate
            # exception handling and shutdown to the existing Jupyter context.
            with ExitStack() as stack:
                with lock_type(str(startup_lock), timeout=300):
                    stack.enter_context(setup_kernel(**kwargs))
                yield

        executor.setup_kernel = locked_setup
    executed, resources = executor.preprocess(notebook)
    if not isinstance(executed, NotebookNode):
        raise TypeError("notebook execution must return a NotebookNode")
    if not isinstance(resources, dict) or not all(
        isinstance(key, str) for key in resources
    ):
        raise TypeError("notebook resources must be a string-keyed dictionary")
    return executed, resources
