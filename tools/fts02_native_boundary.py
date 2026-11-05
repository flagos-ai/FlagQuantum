"""Linux seccomp boundary for the pinned FTS-02 executables."""

from __future__ import annotations

import ctypes
import errno
import os
import sys

ALLOW = 0x7FFF0000
ERRNO = 0x00050000
CLONE_THREAD = 0x00010000
MASKED_EQ = 7


class Comparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("mask", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
    ]


def install() -> None:
    """Deny new process groups at the syscall boundary and allow threads."""
    if sys.platform != "linux":
        raise RuntimeError("FTS-02 containment requires Linux and libseccomp")
    if os.uname().machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("unreviewed architecture for seccomp clone arguments")
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Comparison),
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(ALLOW)
    if not context:
        raise RuntimeError("cannot allocate seccomp filter")
    try:
        for name, error, comparison in (
            (b"fork", errno.EPERM, None),
            (b"vfork", errno.EPERM, None),
            (b"clone3", errno.ENOSYS, None),
            (b"clone", errno.EPERM, Comparison(0, MASKED_EQ, CLONE_THREAD, 0)),
            (b"setsid", errno.EPERM, None),
            (b"setpgid", errno.EPERM, None),
        ):
            number = lib.seccomp_syscall_resolve_name(name)
            if (
                number == -1
                or lib.seccomp_rule_add_array(
                    context,
                    ERRNO | error,
                    number,
                    int(comparison is not None),
                    ctypes.byref(comparison) if comparison is not None else None,
                )
                != 0
            ):
                raise RuntimeError(f"cannot install seccomp rule for {name!r}")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("cannot load seccomp filter")
    finally:
        lib.seccomp_release(context)
