"""Build configuration for FlagQuantum's package-local native operators."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

TORCH_USES_OPENMP = "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()

if os.name == "nt":
    CPP_FLAGS = ["/O2", "/std:c++17", "/Brepro"] + (
        ["/openmp:experimental"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["/Brepro"]
elif sys.platform == "darwin":
    CPP_FLAGS = ["-O3", "-g0", "-std=c++17"] + (
        ["-Xpreprocessor", "-fopenmp"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["-Wl,-rpath,@loader_path/../../../torch/lib"] + (
        ["-lomp"] if TORCH_USES_OPENMP else []
    )
else:
    CPP_FLAGS = ["-O3", "-g0", "-std=c++17"] + (
        ["-fopenmp"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["-Wl,-rpath,$ORIGIN/../../../torch/lib"] + (
        ["-fopenmp"] if TORCH_USES_OPENMP else []
    )

CPP_FLAGS.extend(
    [
        "/DPy_LIMITED_API=0x030A0000",
        "/DTORCH_TARGET_VERSION=0x020A000000000000",
    ]
    if os.name == "nt"
    else [
        "-DPy_LIMITED_API=0x030A0000",
        "-DTORCH_TARGET_VERSION=0x020A000000000000",
    ]
)
if TORCH_USES_OPENMP:
    CPP_FLAGS.append(
        "/DFQ_NATIVE_CPU_PARALLEL=1"
        if os.name == "nt"
        else "-DFQ_NATIVE_CPU_PARALLEL=1"
    )


def _configure_msvc_environment() -> None:
    """Give cl.exe its required flags independently of packaging internals."""

    configured = os.environ.get("CL", "").strip()
    configured_flags = configured.split()
    missing = [flag for flag in CPP_FLAGS if flag not in configured_flags]
    os.environ["CL"] = " ".join(part for part in (configured, *missing) if part)


if os.name == "nt":
    # MSVC reads CL itself and prepends its contents to every compiler command.
    # This public compiler interface is the fallback for PyTorch 2.10's
    # non-Ninja wrapper, whose post-arguments are bypassed by current
    # setuptools' MSVC Compiler implementation.
    _configure_msvc_environment()


def _normalize_macho_uuid(path: Path) -> None:
    """Replace a thin Mach-O UUID with a deterministic content-derived UUID."""

    data = bytearray(path.read_bytes())
    if len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe":
        return
    command_count = struct.unpack_from("<I", data, 16)[0]
    offset = 32
    for _ in range(command_count):
        command, size = struct.unpack_from("<II", data, offset)
        if command == 0x1B and size == 24:
            uuid_offset = offset + 8
            data[uuid_offset : uuid_offset + 16] = b"\0" * 16
            digest = bytearray(hashlib.sha256(data).digest()[:16])
            digest[6] = (digest[6] & 0x0F) | 0x40
            digest[8] = (digest[8] & 0x3F) | 0x80
            data[uuid_offset : uuid_offset + 16] = digest
            path.write_bytes(data)
            return
        offset += size


class ReproducibleBuildExtension(BuildExtension):
    """Keep compiled wheels byte-reproducible on Mach-O platforms."""

    def build_extension(self, ext: Any) -> None:
        super().build_extension(ext)
        if sys.platform == "darwin":
            extension_path = Path(self.get_ext_fullpath(ext.name))
            if TORCH_USES_OPENMP:
                torch_openmp = Path(torch.__file__).parent / "lib" / "libomp.dylib"
                openmp_install_name = subprocess.run(
                    ("otool", "-D", str(torch_openmp)),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()[1]
                subprocess.run(
                    (
                        "install_name_tool",
                        "-change",
                        openmp_install_name,
                        "@rpath/libomp.dylib",
                        str(extension_path),
                    ),
                    check=True,
                )
            _normalize_macho_uuid(extension_path)
            subprocess.run(
                ("codesign", "--force", "--sign", "-", str(extension_path)),
                check=True,
            )


setup(
    ext_modules=[
        CppExtension(
            "flagquantum.simulation.native_cpu._C",
            [
                "flagquantum/simulation/native_cpu/csrc/permutation.cpp",
                "flagquantum/simulation/native_cpu/csrc/rotation_adjoint.cpp",
                "flagquantum/simulation/native_cpu/csrc/module.cpp",
            ],
            # Keep the keyed form as the normal PyTorch forwarding route.  On
            # Windows the CL compiler environment above independently carries
            # the same flags for packaging stacks that bypass post-arguments.
            extra_compile_args={"cxx": CPP_FLAGS},
            extra_link_args=LINK_FLAGS,
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": ReproducibleBuildExtension.with_options(use_ninja=False)},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
