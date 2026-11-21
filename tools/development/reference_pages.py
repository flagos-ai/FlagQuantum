# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_page.py.

"""Internal documentation page records built from optional Griffe objects."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from griffe import Alias, Class, Module, Object


@dataclass
class MajorClassPage:
    """Describe a class rendered on its own reference page."""

    obj: Class | Alias | None = None
    section: str | None = None
    pref_path: str | None = None

    @property
    def section_not_none(self) -> str:
        assert self.section is not None, f"Uninitialized {self}"
        return self.section

    def out_path(self, out_dir: Path) -> Path:
        assert self.pref_path is not None, f"Uninitialized {self}"
        segments = self.pref_path.split(".")
        return out_dir / "/".join(segments[:-1]) / f"{segments[-1]}.md"


class MemberType(Enum):
    MODULE = "module"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class ModulePageMember:
    obj: Object | Alias
    pref_dotpath: str
    mytype: MemberType


@dataclass
class ModulePage:
    """Describe a module page and its ordered member references."""

    obj: Module | Alias | None = None
    section: str | None = None
    pref_path: str | None = None
    members: list[ModulePageMember] = field(default_factory=list)

    @property
    def section_not_none(self) -> str:
        assert self.section is not None, f"Uninitialized {self}"
        return self.section

    def out_path(self, out_dir: Path) -> Path:
        assert self.pref_path is not None, f"Uninitialized {self}"
        segments = self.pref_path.split(".")
        return out_dir / "/".join(segments[:-1]) / f"{segments[-1]}.md"


Page = ModulePage | MajorClassPage
