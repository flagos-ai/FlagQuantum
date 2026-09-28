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
        ["/openmp"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["/Brepro"]
elif sys.platform == "darwin":
    CPP_FLAGS = ["-O3", "-g0", "-std=c++17"] + (
        ["-Xpreprocessor", "-fopenmp"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["-lomp"] if TORCH_USES_OPENMP else []
else:
    CPP_FLAGS = ["-O3", "-g0", "-std=c++17"] + (
        ["-fopenmp"] if TORCH_USES_OPENMP else []
    )
    LINK_FLAGS = ["-Wl,-rpath,$ORIGIN/../../../torch/lib"] + (
        ["-fopenmp"] if TORCH_USES_OPENMP else []
    )


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
            _normalize_macho_uuid(extension_path)
            subprocess.run(
                ("codesign", "--force", "--sign", "-", str(extension_path)),
                check=True,
            )


setup(
    ext_modules=[
        CppExtension(
            "flagquantum.simulation.native_cpu._C",
            ["flagquantum/simulation/native_cpu/csrc/rotation_adjoint.cpp"],
            extra_compile_args=CPP_FLAGS,
            extra_link_args=LINK_FLAGS,
        )
    ],
    cmdclass={"build_ext": ReproducibleBuildExtension.with_options(use_ninja=False)},
)
