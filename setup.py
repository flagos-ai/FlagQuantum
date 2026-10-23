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


def _torch_library_directory() -> Path:
    """Return the directory that holds the installed PyTorch shared libraries."""

    return Path(torch.__file__).parent / "lib"


def _absolute_rpath_flag(directory: Path) -> str:
    """Return the linker flag that adds one absolute library directory."""

    return f"-Wl,-rpath,{directory}"


#: Marks the throwaway environment a build frontend installs the build
#: requirements into. `python -m pip install -e .` without
#: `--no-build-isolation` builds in `.../pip-build-env-XXXX/overlay/...` under
#: the system temporary directory and deletes it once the install finishes.
BUILD_ENVIRONMENT_MARKER = "build-env"


def _is_build_environment_directory(directory: Path) -> bool:
    """Return whether a directory belongs to a frontend's throwaway environment."""

    return any(BUILD_ENVIRONMENT_MARKER in part for part in directory.parts)


class ReproducibleBuildExtension(BuildExtension):
    """Make the compiled operator loadable, and keep a wheel byte-reproducible.

    A distributed wheel must not embed this machine's PyTorch path, so the
    relocatable rpath stays the only one it carries. An in-place build lands in
    ``<source>/flagquantum/simulation/native_cpu/``, where that relocatable
    rpath resolves to ``<source>/torch/lib`` -- a directory that does not
    exist, so the artifact cannot be loaded at all. The absolute directory is
    therefore added only to an in-place build, which is exactly the build whose
    final location is already known, and the artifact that build delivers into
    the source tree is then opened, so a build that cannot produce a loadable
    operator fails instead of returning success.
    """

    inplace_build = False

    def finalize_options(self) -> None:
        super().finalize_options()
        # `inplace` is set here and cleared again before `build_extension` runs,
        # so both the link decision and the load check record it now.
        self.inplace_build = bool(self.inplace)
        if not self.inplace_build or os.name == "nt":
            return
        directory = _torch_library_directory()
        self._require_durable_rpath(directory)
        flag = _absolute_rpath_flag(directory)
        for extension in self.extensions:
            extension.extra_link_args = [*(extension.extra_link_args or ()), flag]

    def _require_durable_rpath(self, directory: Path) -> None:
        """Fail when the artifact would depend on a directory pip deletes.

        An isolated build installs its build requirements into a throwaway
        environment under the system temporary directory, so ``torch.__file__``
        names that copy and the absolute library directory recorded here is
        gone once the install finishes. The load check in `run` cannot see it:
        it opens the artifact while the environment still exists, so it passes
        and the artifact stops opening afterwards.

        macOS is where that silent loss was measured, because it is the one
        dependency that cannot survive it. PyTorch's own libraries are already
        loaded underneath `_C` and their install names are ``@rpath/libc10.dylib``
        and friends, so dyld matches them without any rpath. The OpenMP runtime
        is the exception: torch ships it with a build-machine install name
        (``/opt/llvm-openmp/lib/libomp.dylib``), the reference is rewritten to
        ``@rpath/libomp.dylib`` so it can be relocated at all, and that string
        matches nothing already open -- it resolves only through an rpath, which
        is the one this build is about to record into a deleted directory.
        Other platforms are not claimed to be unaffected; this check reports
        only the failure it was measured against.
        """

        if sys.platform != "darwin" or not _is_build_environment_directory(directory):
            return
        raise RuntimeError(
            "an editable build cannot record a library search path from an "
            f"isolated build environment: {directory} belongs to the build "
            "frontend and is deleted when the install finishes, so the operator "
            "left in the source tree would stop opening and every native CPU "
            "path would silently fall back to PyTorch. Install with "
            "`python -m pip install --no-build-isolation -e .` as "
            "CONTRIBUTING.md documents, or build a wheel, which locates the "
            "libraries relative to the installed package."
        )

    def run(self) -> None:
        super().run()
        if not self.inplace_build:
            return
        # setuptools restores `inplace` once the staged build is done and copies
        # the extensions into the source tree, so the path resolved here is the
        # one the runtime loads.
        for extension in self.extensions:
            self._require_loadable_artifact(extension)

    def build_extension(self, ext: Any) -> None:
        if self.inplace_build:
            # A staged artifact is reused whenever it is newer than its sources,
            # and the link command is not part of that comparison, so an
            # extension linked by an earlier build would keep the rpath that
            # build chose. Removing only the linked output forces the link this
            # build configured and leaves the object files to be reused.
            Path(self.get_ext_fullpath(ext.name)).unlink(missing_ok=True)
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

    def _require_loadable_artifact(self, ext: Any) -> None:
        """Fail the build when the artifact delivered to the source tree cannot be opened.

        ``native_cpu.adjoint._load_extension`` reports a failed ``dlopen`` as
        ``False`` so that an unbuilt extension degrades to the Python fallback
        instead of raising. A built extension that cannot load is a different
        state that must not borrow that silence: it means every native path is
        silently running the fallback while the build reports success.
        """

        artifact = Path(self.get_ext_fullpath(ext.name)).resolve()
        # The runtime imports torch before the extension, and on Windows that
        # import is what puts the PyTorch DLLs on the search path, so the probe
        # opens the artifact the way the runtime does.
        probe = (
            "import importlib.util, sys\n"
            "import torch\n"
            "spec = importlib.util.spec_from_file_location('_C', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
        )
        completed = subprocess.run(
            (sys.executable, "-c", probe, str(artifact)),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{artifact} was built but cannot be loaded:\n"
                f"{completed.stderr.strip()}"
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
