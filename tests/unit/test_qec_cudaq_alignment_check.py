#!/usr/bin/env python3
"""Mutation test for the QEC <-> CUDA-Q alignment checklist checker.

A checker that only ever prints OK proves nothing. This test damages one section
of the checklist at a time, runs the checker, and requires the run to fail for
the reason the damage was supposed to cause. It also plants a fake module in a
throwaway copy of the repository to prove the symbol index really reads the tree
rather than pattern-matching the checklist against itself.

Mutating a file and running a tool is a subprocess-shaped proof, so it does not
live in a checker: it lives here, where the default lane runs it and where a
failure names the mutation. The unit marker is honest -- no network, no device,
no accelerator, no cluster -- but the test is not cheap, because the checker is
only meaningful when it is reading a real repository. One copy of this tree is
made per module and reused by both tests.

Mutations are addressed structurally -- by row id and key name -- rather than by
quoting a long literal, so editing the prose of a row cannot silently turn a
mutation into a no-op. Every mutation is applied to a temporary copy; the
checklist and the repository are never modified in place.

Run directly for the mutation-by-mutation log:

    python -m pytest tests/unit/test_qec_cudaq_alignment_check.py -q -s
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools/check_qec_cudaq_alignment.py"
CHECKLIST = ROOT / "contracts/qec-cudaq-alignment-checklist.toml"
REPO_SOURCE = ROOT

# Copied into the throwaway tree so the checklist can claim a name the real tree
# lacks and then gain it, which is the only way to show that absence is measured
# rather than assumed.
PLANTED_MODULE = "flagquantum/qec/planted.py"
PLANTED_SYMBOL = "flagquantum.qec.PlantedSymbol"

# Ignored when copying the repository: none of these can change the answer, and
# the history alone is larger than everything the checker reads.
_SKIP = shutil.ignore_patterns(
    ".git", "__pycache__", ".venv", "*.egg-info", ".pytest_cache"
)


# --------------------------------------------------------------------------
# structural editing of the checklist
# --------------------------------------------------------------------------


def block_span(lines: list[str], row_id: str) -> tuple[int, int]:
    """Line span of one `[[...]]` block, from its header to the next header."""

    marker = f'id = "{row_id}"'
    try:
        anchor = next(i for i, line in enumerate(lines) if line.strip() == marker)
    except StopIteration:
        raise KeyError(f"no row with id {row_id!r}") from None
    start = anchor
    while start > 0 and not lines[start].startswith("[["):
        start -= 1
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("[["):
            end = i
            break
    return start, end


def set_key(text: str, row_id: str, key: str, value: str | None) -> str:
    """Set (or with value None, rename away) a key inside one row block."""

    lines = text.splitlines(keepends=True)
    start, end = block_span(lines, row_id)
    for i in range(start, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key} = ") or stripped.startswith(f"{key} =["):
            if value is None:
                # Rename rather than delete: the checker must notice a missing
                # key, not a syntactically broken file.
                indent = lines[i][: len(lines[i]) - len(stripped)]
                lines[i] = f"{indent}{key}_renamed{stripped[len(key) :]}"
            else:
                indent = lines[i][: len(lines[i]) - len(stripped)]
                lines[i] = f"{indent}{value}\n"
            return "".join(lines)
    raise KeyError(f"row {row_id!r} has no {key!r} key")


def set_header_key(text: str, key: str, value: str) -> str:
    """Replace a top-level header key, which lives before the first block."""

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(f"{key} = "):
            lines[i] = f"{value}\n"
            return "".join(lines)
    raise KeyError(f"the header has no {key!r} key")


def append_to_list(text: str, row_id: str, key: str, item: str) -> str:
    """Append one entry to a single-line or multi-line list inside a row."""

    lines = text.splitlines(keepends=True)
    start, end = block_span(lines, row_id)
    for i in range(start, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key} = ["):
            indent = lines[i][: len(lines[i]) - len(stripped)]
            if stripped.rstrip().endswith("]"):
                body = stripped.rstrip()[:-1].rstrip()
                sep = "" if body.endswith("[") else ", "
                lines[i] = f"{indent}{body}{sep}{item}]\n"
            else:
                lines.insert(i + 1, f'{indent}    "{item}",\n')
            return "".join(lines)
    raise KeyError(f"row {row_id!r} has no {key!r} list")


def sub_in_key(text: str, row_id: str, key: str, old: str, new: str) -> str:
    """Replace a substring inside one key's value, refusing a no-op.

    The value may span several lines, so the whole block is searched rather than
    only the line the key starts on.
    """

    lines = text.splitlines(keepends=True)
    start, end = block_span(lines, row_id)
    key_at = None
    for i in range(start, end):
        if lines[i].lstrip().startswith(f"{key} ="):
            key_at = i
            break
    if key_at is None:
        raise KeyError(f"row {row_id!r} has no {key!r} key")
    # The value runs to the end of the block, or to the next top-level key.
    stop = end
    for i in range(key_at + 1, end):
        if lines[i] and not lines[i][0].isspace() and "=" in lines[i]:
            stop = i
            break
    value = "".join(lines[key_at:stop])
    if old not in value:
        raise KeyError(f"row {row_id!r} has no {key!r} containing {old!r}")
    lines[key_at:stop] = [value.replace(old, new, 1)]
    return "".join(lines)


# --------------------------------------------------------------------------
# the mutations
# --------------------------------------------------------------------------

# (name, expected failure substring, mutation function)
MUTATIONS: list[tuple[str, str, Callable[[str], str]]] = [
    (
        "delete a symbol a row depends on",
        "does not resolve",
        lambda t: sub_in_key(
            t,
            "qec_dem_text_interchange",
            "symbols_present",
            "to_stim_text",
            "to_stim_text_typo",
        ),
    ),
    (
        "claim a gap the tree has already filled",
        "exists as a definition somewhere",
        lambda t: append_to_list(
            t, "qec_dem_construction", "symbols_absent", "DetectorErrorModel"
        ),
    ),
    (
        "point evidence at a file that does not exist",
        "evidence path",
        lambda t: sub_in_key(
            t, "qec_code_record", "evidence", "codes.py", "codes_missing.py"
        ),
    ),
    (
        "claim a maturity entry that is not registered",
        "maturity_ref",
        lambda t: set_key(
            t, "qec_code_record", "maturity_ref", 'maturity_ref = "not_a_capability"'
        ),
    ),
    (
        "drop the maturity_ref from a supported row",
        "maturity_ref",
        lambda t: set_key(t, "qec_code_record", "maturity_ref", None),
    ),
    (
        "point a row at a capability its matrix row does not name",
        "is not one the parity matrix row",
        lambda t: set_key(
            t,
            "qec_decoder_family",
            "maturity_ref",
            'maturity_ref = "detector_error_model"',
        ),
    ),
    (
        "point the migration row at the sampling half of Stim integration",
        "is not one the parity matrix row",
        lambda t: set_key(
            t,
            "qec_stim_user_migration",
            "maturity_ref",
            'maturity_ref = "stabilizer_sampling"',
        ),
    ),
    (
        "claim a component version the parity contract does not record",
        "is not the component version the parity contract records",
        lambda t: set_header_key(
            t, "baseline_component_version", 'baseline_component_version = "0.7.0"'
        ),
    ),
    (
        "quote a component release index the parity contract does not record",
        "does not name the release index the parity contract records",
        lambda t: set_header_key(
            t,
            "baseline_component_source",
            'baseline_component_source = "https://pypi.org/pypi/stim/json"',
        ),
    ),
    (
        "invent a parity matrix row",
        "is not a capability of the quantum_error_correction domain",
        lambda t: set_key(
            t, "qec_dialect", "domain_row", 'domain_row = "qec_dialect_typo"'
        ),
    ),
    (
        "quote a CUDA-Q surface item nobody recorded",
        "is not a recorded CUDA-Q surface item",
        lambda t: set_key(
            t,
            "qec_dialect",
            "domain_items",
            'domain_items = ["an invented QEC dialect"]',
        ),
    ),
    (
        "override the matrix priority without stating why",
        "floor_override_reason",
        lambda t: sub_in_key(
            t,
            "qec_dem_chunking",
            "floor_override_reason",
            "floor_override_reason = ",
            "floor_override_reason_x = ",
        ),
    ),
    (
        "drop the floor_override_reason where no matrix row backs the floor",
        "floor_override_reason",
        lambda t: sub_in_key(
            t,
            "qec_dem_matrices_and_rates",
            "floor_override_reason",
            "floor_override_reason = ",
            "floor_override_reason_x = ",
        ),
    ),
    (
        "drop the fail-closed statement from a row",
        "fail_closed is empty",
        lambda t: set_key(t, "qec_dialect", "fail_closed", 'fail_closed = ""'),
    ),
    (
        "drop the next_action from a partial row",
        "next_action is empty",
        lambda t: set_key(
            t, "qec_stim_sampling_join", "next_action", 'next_action = ""'
        ),
    ),
    (
        # An aligned row's bar is the presence of something to point at, and a
        # row that clears the bar for every other status must still be rejected
        # when it names nothing. This is the only mutation that reaches the
        # aligned branch, so it is also the evidence that the branch exists.
        "claim alignment while pointing at nothing",
        "an `aligned` row must name at least one symbols_present entry",
        lambda t: set_key(t, "qec_dem_merge", "symbols_present", None),
    ),
    (
        # Alignment is a claim about a row's own surface, not a licence to stop
        # pointing at it: the row this is planted in is the one the handle layer
        # closed, and an aligned row is exactly where a symbol that stopped
        # resolving would otherwise go unnoticed.
        "stop a handle symbol resolving in a row that claims alignment",
        "does not resolve",
        lambda t: sub_in_key(
            t,
            "qec_detector_annotations",
            "symbols_present",
            "flagquantum.qec.MeasurementSamples",
            "flagquantum.qec.MeasurementSamplesGone",
        ),
    ),
    (
        # An absence is a live claim, so the day the symbol it names exists the
        # row is stale rather than merely out of date: this is the mutation that
        # fails if the handle layer is ever deleted while the row keeps saying
        # the layer is absent.
        "keep an absence standing after the handle layer closed it",
        "now resolves -- the gap closed, so this row is stale",
        lambda t: append_to_list(
            t,
            "qec_detector_annotations",
            "symbols_absent",
            "flagquantum.qec.MeasurementSamples",
        ),
    ),
    (
        # `next_action` is required of a partial or absent row and of no other,
        # so this is the other half of that rule: an aligned row may drop it,
        # and must still state what it is for.
        "leave a row that dropped its next_action without a target",
        "target is empty",
        lambda t: set_key(t, "qec_detector_annotations", "target", 'target = ""'),
    ),
    (
        # The row this mutation is planted in is `partial` on the other half of
        # its own scope, so the branch it reaches is the partial one. It moved
        # here when `qec_dem_matrices_and_rates` became `aligned`: a row that
        # changes status must not quietly move a mutation onto the aligned
        # branch, where it would still fail but for a different reason.
        "leave a partial row with nothing present",
        "must name the symbols_present entry",
        lambda t: set_key(t, "qec_dem_construction", "symbols_present", None),
    ),
    (
        "leave a gap row with no proof at all",
        "must prove the gap",
        lambda t: set_key(
            set_key(t, "qec_dem_chunking", "symbols_absent", None),
            "qec_dem_chunking",
            "negative_search",
            None,
        ),
    ),
    (
        "claim a gap whose proof path now exists",
        "now exists",
        lambda t: append_to_list(
            t, "qec_dem_chunking", "negative_search", "flagquantum/qec/dem.py"
        ),
    ),
    (
        "claim a status the parity matrix contradicts",
        "while parity matrix row",
        lambda t: set_key(t, "qec_dialect", "status", 'status = "aligned"'),
    ),
    (
        "mark a diff row absent while still naming a symbol",
        "verdict `absent`",
        lambda t: set_key(
            t, "dem_measurement_to_detector_map", "verdict", 'verdict = "absent"'
        ),
    ),
    (
        "leave a diff row without a note",
        "note is empty",
        lambda t: set_key(t, "dem_detector_matrix", "note", 'note = ""'),
    ),
    (
        "leave a diff row without a CUDA-Q symbol",
        "cudaq_symbol is empty",
        lambda t: set_key(t, "dem_error_ids", "cudaq_symbol", 'cudaq_symbol = ""'),
    ),
    (
        # The mark is planted rather than borrowed from a row that already
        # carries it, because the contract ships with none: every upstream
        # negative in it is now pinned to the revision it was taken at. A
        # mutation that took the mark off an existing row would have stopped
        # testing the rule the day the last row was settled, and would have gone
        # on passing while testing nothing if a row dropped the mark without
        # saying so.
        "mark a row unverified without saying so in the row",
        "does not say UNVERIFIED",
        lambda t: sub_in_key(
            t,
            "dem_carrier",
            "verdict",
            'verdict = "reshaped"',
            'verdict = "reshaped"\nprovenance_unverified = true',
        ),
    ),
    (
        "hide a row from the reading document",
        "is not mentioned",
        lambda t: set_key(
            t, "dem_canonicalize", "id", 'id = "dem_canonicalize_renamed"'
        ),
    ),
    (
        "name an absent verdict for a symbol the tree has",
        "verdict `absent` but",
        lambda t: set_key(
            t,
            "dem_canonicalize",
            "flagquantum_symbol",
            'flagquantum_symbol = "flagquantum.qec.DetectorErrorModel"',
        ),
    ),
    (
        # A diff row that names a real definition under the wrong owner is the
        # tempting way to claim a merge surface this repository does not have:
        # the last segment is a name the tree defines, so only following the
        # dotted path to its owner separates a claim from a wish.
        "point a diff row at a real name under the wrong owner",
        "'DemMergeRule' exists in the tree but not at",
        lambda t: set_key(
            t,
            "dem_merge_operation",
            "flagquantum_symbol",
            'flagquantum_symbol = "flagquantum.qec.DetectorErrorModel.DemMergeRule"',
        ),
    ),
    (
        "keep a negative_search proof for a gap the tree has closed",
        "now resolves -- the diff is stale",
        lambda t: append_to_list(
            t,
            "dem_merge_operation",
            "negative_search",
            "symbol:flagquantum.qec.DetectorErrorModel.merge_duplicate_mechanisms",
        ),
    ),
    (
        # The registry is the newest diff row to gain a symbol, so the wrong-owner
        # mistake has to be caught here too: `get_decoder` is a real name in the
        # tree, and only following the dotted path separates the registry's
        # factory from a method of the model.
        "point the registry diff row at a real name under the wrong owner",
        "'get_decoder' exists in the tree but not at",
        lambda t: set_key(
            t,
            "decoder_registry",
            "flagquantum_symbol",
            'flagquantum_symbol = "flagquantum.qec.DetectorErrorModel.get_decoder"',
        ),
    ),
    (
        "keep the registry diff row absent now that the tree has it",
        "verdict `absent` but",
        lambda t: set_key(t, "decoder_registry", "verdict", 'verdict = "absent"'),
    ),
    (
        # `error_id` is a field of the mechanism record, not of the model that
        # holds the records, and the two names sit in one module. Only following
        # the dotted path separates the mechanism's own id from a model-level
        # one.
        "point the error-id diff row at a real name under the wrong owner",
        "'error_id' exists in the tree but not at",
        lambda t: set_key(
            t,
            "dem_error_ids",
            "flagquantum_symbol",
            'flagquantum_symbol = "flagquantum.qec.DetectorErrorModel.error_id"',
        ),
    ),
    (
        # The column this looks for has landed, so listing it as absent is how a
        # closed gap keeps being reported as open -- which is what the staleness
        # check reads the tree to catch. The name is planted in the row that
        # consumes the column rather than in the row whose surface it closed,
        # because the check resolves the name against the checkout and does not
        # care which row is doing the reporting. The item is bare because this
        # row's list is written one entry per line, and the mutator supplies the
        # quotes itself for that shape.
        "keep the error-id column listed as absent now that the model states it",
        "symbols_absent 'error_ids' exists as a definition somewhere",
        lambda t: append_to_list(
            t,
            "qec_decoder_family",
            "symbols_absent",
            "error_ids",
        ),
    ),
]


def run(checker: Path, checklist: Path, repo_root: Path) -> tuple[int, str]:
    result = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--checklist",
            str(checklist),
            "--repo-root",
            str(repo_root),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


@dataclass
class Sandbox:
    """A throwaway repository copy and the checklist the checker is pointed at.

    The checklist is rewritten before every run, so a test reads as a list of
    damages and the repository copy stays pristine.
    """

    repo_root: Path
    checklist: Path

    def check(self, checklist_text: str) -> tuple[int, str]:
        self.checklist.write_text(checklist_text, encoding="utf-8")
        return run(CHECKER, self.checklist, self.repo_root)

    def settle(self, name: str, expected: str, checklist_text: str) -> None:
        """Require the checker to reject one damaged checklist for one reason."""

        code, output = self.check(checklist_text)
        assert code != 0, f"{name}: the checker accepted a damaged checklist"
        assert expected in output, (
            f"{name}: the checker failed for the wrong reason; wanted {expected!r}, "
            f"it said:\n{output}"
        )


@pytest.fixture(scope="module")
def sandbox() -> Iterator[Sandbox]:
    """One copy of this repository, because the checker has to read a real tree."""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_root = tmp_path / "fq"
        shutil.copytree(REPO_SOURCE, repo_root, symlinks=True, ignore=_SKIP)
        yield Sandbox(repo_root=repo_root, checklist=tmp_path / "checklist.toml")


@pytest.fixture(scope="module")
def baseline() -> str:
    return CHECKLIST.read_text(encoding="utf-8")


def test_the_committed_checklist_passes_and_the_index_reads_the_tree(
    sandbox: Sandbox, baseline: str
) -> None:
    """The baseline passes, and it passes for a measurable reason.

    A checker could satisfy the baseline by agreeing with the checklist rather
    than by reading the repository, so three controls separate the two: a
    definition that exists only in the tree must be seen, a name the tree lacks
    must be rejected, and the planted claim must pass again once the definition
    is removed.
    """

    code, output = sandbox.check(baseline)
    assert code == 0, f"the committed checklist should pass, but it said:\n{output}"

    # A definition that exists only in the source tree, not in the checklist.
    planted = sandbox.repo_root / PLANTED_MODULE
    planted.write_text("class PlantedSymbol:\n    pass\n", encoding="utf-8")
    sandbox.settle(
        "planted symbol",
        f"{PLANTED_SYMBOL.rpartition('.')[2]!r} exists as a definition somewhere",
        append_to_list(
            baseline, "qec_dem_construction", "symbols_absent", "PlantedSymbol"
        ),
    )

    # A positive claim for a name nothing in the tree defines.
    sandbox.settle(
        "resolves nowhere",
        "does not resolve",
        append_to_list(
            baseline,
            "qec_code_record",
            "symbols_present",
            "flagquantum.qec.NotInTheTree",
        ),
    )

    # With the module gone, the same claim must fail again. An index that read
    # the checklist instead of the tree could not make that distinction.
    planted.unlink()
    code, output = sandbox.check(
        append_to_list(
            baseline, "qec_dem_construction", "symbols_absent", "PlantedSymbol"
        )
    )
    assert code == 0, (
        "the planted claim still failed after the module was removed, so the "
        f"earlier pass was not caused by reading the tree:\n{output}"
    )


def test_every_mutation_is_caught_for_its_own_reason(
    sandbox: Sandbox, baseline: str
) -> None:
    """Each damage must be caught, and caught for the reason it was designed for.

    A mutation that is caught for the wrong reason is a mutation that would keep
    passing if the check it targets were deleted, so the expected substring is
    part of the assertion rather than a note.
    """

    failures: list[str] = []
    caught = 0
    for name, expected, mutate in MUTATIONS:
        try:
            damaged = mutate(baseline)
        except KeyError as error:
            failures.append(f"{name}: could not be applied ({error})")
            continue
        if damaged == baseline:
            failures.append(f"{name}: the mutation was a no-op, so it proves nothing")
            continue
        try:
            sandbox.settle(name, expected, damaged)
        except AssertionError as error:
            failures.append(str(error))
            continue
        caught += 1

    assert not failures, (
        f"{len(failures)} of {len(MUTATIONS)} mutations did not behave as required:\n  - "
        + "\n  - ".join(failures)
    )
    assert caught == len(MUTATIONS)
