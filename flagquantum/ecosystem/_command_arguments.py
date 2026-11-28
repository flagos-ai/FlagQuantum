# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/arg_parse.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""First-match command flags with explicit source scalar and file ownership."""

import ctypes
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from math import copysign, isnan
from typing import BinaryIO, Literal, TypeVar, cast

from ._command_numbers import parse_c_float, parse_int64

_T = TypeVar("_T")


def _float_text(value: float) -> str:
    return "-nan" if isnan(value) and copysign(1, value) < 0 else format(value, ".6g")


def find_argument(name: str, argv: Sequence[str]) -> str | None:
    name = name.partition("\0")[0]
    arguments = tuple(word.partition("\0")[0] for word in argv)
    for index in range(1, len(arguments)):
        word = arguments[index]
        if word == "--":
            break
        if word.startswith(name + "="):
            return word[len(name) + 1 :]
        if word == name:
            if index == len(arguments) - 1:
                return ""
            following = arguments[index + 1]
            if following.startswith("-") and not (
                len(following) > 1 and "0" <= following[1] <= "9"
            ):
                return ""
            return following
    return None


def require_argument(name: str, argv: Sequence[str]) -> str:
    result = find_argument(name, argv)
    if result is None:
        raise ValueError(f"\033[31mMissing command line argument: '{name}'")
    return result


def check_unknown_arguments(
    known: Sequence[str],
    deprecated: Sequence[str],
    program: str,
    mode: str | None,
    argv: Sequence[str],
) -> None:
    arguments = tuple(word.partition("\0")[0] for word in argv)
    index = 1
    while index < len(arguments):
        if mode is not None and index == 1 and arguments[index] == mode:
            index += 1
            continue
        if arguments[index] == "--":
            break
        matched = False
        for names in (known, deprecated):
            for name in names:
                word = arguments[index]
                if word == name or word.startswith(name + "="):
                    if (
                        word == name
                        and index + 1 < len(arguments)
                        and not arguments[index + 1].startswith("-")
                    ):
                        index += 1
                    matched = True
                    break
        if not matched:
            suffix = "" if mode is None else f" for `{program} {mode}`"
            recognized = "".join(f"    {name}\n" for name in sorted(set(known)))
            raise ValueError(
                f"Unrecognized command line argument {arguments[index]}{suffix}.\n"
                f"Recognized command line arguments{suffix}:\n{recognized}"
            )
        index += 1


def bool_argument(name: str, argv: Sequence[str]) -> bool:
    text = find_argument(name, argv)
    if text:
        raise ValueError(f"Got non-empty value '{text}' for boolean flag '{name}'.")
    return text is not None


def int64_argument(
    name: str, default: int, minimum: int, maximum: int, argv: Sequence[str]
) -> int:
    text = find_argument(name, argv)
    if not text:
        if not minimum <= default <= maximum:
            raise ValueError(f"Must specify a value for int flag '{name}'.")
        return default
    try:
        value = parse_int64(text)
    except ValueError as error:
        raise ValueError(
            f"Got non-int64 value '{text}' for int64 flag '{name}'."
        ) from error
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Integer value '{text}' for flag '{name}' doesn't satisfy {minimum} <= {value} <= {maximum}."
        )
    return value


def float_argument(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    argv: Sequence[str],
    *,
    precision: Literal[32, 64] = 64,
) -> float:
    # C++ converts scalar defaults/bounds to the selected overload on entry.
    if precision == 32:
        from ctypes import c_float

        default, minimum, maximum = (
            c_float(value).value for value in (default, minimum, maximum)
        )
    text = find_argument(name, argv)
    if text is None:
        if default < minimum or default > maximum:
            raise ValueError(f"Must specify a value for float flag '{name}'.")
        return default
    try:
        value = parse_c_float(text, precision)
    except ValueError as error:
        kind = "float" if precision == 32 else "double-precision-float"
        label = "float flag" if precision == 32 else "flag"
        raise ValueError(
            f"Got non-{kind} value '{text}' for {label} '{name}'."
        ) from error
    if value < minimum or value > maximum or isnan(value):
        kind = "Float" if precision == 32 else "Double precision float"
        raise ValueError(
            f"{kind} value '{text}' for flag '{name}' doesn't satisfy "
            f"{_float_text(minimum)} <= {_float_text(value)} <= {_float_text(maximum)}."
        )
    return value


def enum_argument(
    name: str, default: str | None, values: Mapping[str, _T], argv: Sequence[str]
) -> _T:
    text = find_argument(name, argv)
    if text is None:
        if default is None:
            raise ValueError(
                f"\033[31mMust specify a value for enum flag '{name}'.\n\033[0m"
            )
        return values[default]
    if text not in values:
        choices = "".join(
            f"    '{key}'" + (" (default)" if key == default else "") + "\n"
            for key in sorted(values)
        )
        raise ValueError(
            f"\033[31mUnrecognized value '{text}' for enum flag '{name}'.\nRecognized values are:\n{choices}\033[0m"
        )
    return values[text]


@contextmanager
def open_file_argument(
    name: str, default: BinaryIO | None, mode: str, argv: Sequence[str]
) -> Iterator[BinaryIO]:
    path = find_argument(name, argv)
    if path is None:
        if default is None:
            raise ValueError(f"Missing command line argument: '{name}'")
        yield default
        return
    if not path:
        raise ValueError(
            f"Command line argument '{name}' can't be empty. It's supposed to be a file path."
        )
    binary_modes: dict[
        str, Literal["rb", "wb", "ab", "rb+", "wb+", "ab+", "xb", "xb+"]
    ] = {
        "r": "rb",
        "w": "wb",
        "a": "ab",
        "r+": "rb+",
        "w+": "wb+",
        "a+": "ab+",
        "wx": "xb",
        "w+x": "xb+",
        "wx+": "xb+",
    }
    normalized = mode.replace("b", "")
    try:
        if normalized in binary_modes:
            stream = open(path, binary_modes[normalized])
        else:
            stream = _open_host_c_file(path, mode)
    except OSError as error:
        raise ValueError(f"Failed to open '{path}'") from error

    try:
        yield stream
    finally:
        stream.close()


def _open_host_c_file(path: str, mode: str) -> BinaryIO:
    """Transfer a newly acquired POSIX C stream to an owned Python descriptor.

    Host fopen determines mode admission and file effects. No C reads or writes
    occur before transfer, so the duplicate retains the exact initial position.
    """
    if os.name != "posix":
        raise ValueError(f"Unsupported C file mode {mode!r}")
    import fcntl

    runtime = ctypes.CDLL(None)
    runtime.fopen.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
    runtime.fopen.restype = ctypes.c_void_p
    runtime.fileno.argtypes = (ctypes.c_void_p,)
    runtime.fileno.restype = ctypes.c_int
    runtime.fclose.argtypes = (ctypes.c_void_p,)
    runtime.fclose.restype = ctypes.c_int
    handle = runtime.fopen(os.fsencode(path), os.fsencode(mode))
    if not handle:
        raise OSError(f"Host fopen refused {path!r}")
    try:
        descriptor = os.dup(runtime.fileno(handle))
    finally:
        runtime.fclose(handle)
    try:
        access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        binary_mode = (
            "rb+" if access == os.O_RDWR else "wb" if access == os.O_WRONLY else "rb"
        )
        return cast(BinaryIO, os.fdopen(descriptor, binary_mode))
    except BaseException:
        os.close(descriptor)
        raise
