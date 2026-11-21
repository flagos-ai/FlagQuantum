# Adapted from Qualtran v0.7.0 make_reference_docs/_pages/render_major_class.py.
#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path

import griffe
from griffe import Kind

from tools.development.files import WriteIfDifferent
from tools.development.reference_context import RenderContext
from tools.development.reference_docstrings import (
    split_docstring,
    write_docstring_parts,
)
from tools.development.reference_links import LinkingWriter, Writable, get_aliases_str
from tools.development.reference_pages import MajorClassPage
from tools.development.reference_signatures import write_method_signature


def write_major_class_member(
    f: Writable, obj: griffe.Class, obj2: griffe.Object | griffe.Alias
) -> None:
    if obj2.is_private:
        return
    if obj2.name == "__init__":
        return
    if obj2.is_special:
        return
    if obj2.kind is Kind.ATTRIBUTE:
        return

    # Member name
    f.write(f"## `{obj2.name}`\n")

    # First docstring line
    first_line, rest = split_docstring(obj2)
    f.write(first_line)
    f.write("\n\n")

    # Optional: signature
    if obj2.kind is Kind.FUNCTION:
        assert isinstance(obj2, (griffe.Function, griffe.Alias))
        write_method_signature(f, obj, obj2)

    # Rest of docstring
    write_docstring_parts(f, rest, level=0)


def write_major_class(
    f: LinkingWriter, obj: griffe.Class, pref_dotpath: str, aliases_d: dict[str, str]
) -> None:
    # Title
    f.write(f"# {obj.name}\n\n")

    # First line of docstring
    first_line, rest = split_docstring(obj)
    f.write(first_line)
    f.write("\n\n")

    # Aliases
    f.write_nl(get_aliases_str(pref_dotpath, aliases_d=aliases_d))
    f.write_nl("\n\n")

    # Rest of docstring
    if rest:
        f.write("## Overview\n")  # Note: must include <h2> before <h3> appears.
        write_docstring_parts(f, rest, level=0)

    # Class members
    for obj2 in obj.members.values():
        write_major_class_member(f, obj, obj2)


def render_major_class(
    base_dir: Path, page: MajorClassPage, render_context: RenderContext
) -> None:
    assert page.pref_path is not None, f"Uninitialized {page}"
    segments = page.pref_path.split(".")
    out_path = base_dir / "/".join(segments[:-1]) / f"{segments[-1]}.md"
    print(f"Writing {page.pref_path} to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with WriteIfDifferent(out_path) as f:
        f2 = render_context.get_linking_writer(f)
        assert page.obj is not None, f"Uninitialized {page}"
        write_major_class(
            f2, page.obj, page.pref_path, aliases_d=render_context.aliases_d
        )
        f2.write_link_targets()
