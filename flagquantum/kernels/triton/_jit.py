"""Typed view of :func:`triton.jit` for the kernel modules in this package.

Triton ships no type information, so a kernel decorated with ``triton.jit``
reads as untyped to a strict checker: ``mypy --strict`` reports every such
kernel as ``untyped-decorator`` and the kernel's own signature is erased.

This module is the type-checker's view of that decorator. It is never
imported at runtime: each kernel module binds ``jit`` to ``triton.jit`` in an
``else`` branch, because the boundary tests in ``tests/unit`` exec a kernel
file directly with a stubbed ``triton`` module, where a relative import has no
parent package to resolve against. Both call forms used by this package are
declared:

    @jit
    def kernel(...): ...

    @jit(do_not_specialize=["bit_position"])
    def kernel(...): ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, overload

import triton

__all__ = ["Kernel", "jit"]


class Kernel(Protocol):
    """The two ways a ``triton.jit`` function is used.

    A kernel is launched by subscripting a grid; a device function is inlined
    by calling it directly. Triton models neither result, so both come back
    as ``Any`` -- which is still more than an untyped decorator allowed, since
    it keeps the decorated name from being ``Any`` itself.
    """

    def __getitem__(self, grid: Any, /) -> Callable[..., Any]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@overload
def jit(fn: Callable[..., Any], /) -> Kernel: ...


@overload
def jit(
    *,
    do_not_specialize: Sequence[str | int] = (),
    **options: Any,
) -> Callable[[Callable[..., Any]], Kernel]: ...


def jit(fn: Any = None, /, **options: Any) -> Any:
    """Decorate a kernel exactly as :func:`triton.jit` does.

    Args:
        fn: The kernel function for the bare ``@jit`` form, or ``None`` when
            the decorator is called with options.
        **options: Keyword options forwarded verbatim, such as
            ``do_not_specialize``.

    Returns:
        The ``triton.JITFunction`` for the bare form, or a decorator returning
        it when options were supplied.
    """
    if fn is None:
        return triton.jit(**options)
    return triton.jit(fn, **options)
