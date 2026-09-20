from __future__ import annotations

import ast
import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(
    r"^```python\n(.*?)^```",
    flags=re.MULTILINE | re.DOTALL,
)
PRINT_NAME = "print"
ALGORITHMS_GUIDE = "docs/guides/ALGORITHMS.md"
# The guide quotes a print statement's output in the comment lines below it, in the
# form `# <value>  -- <prose>`; this is the separator between the value and the prose.
QUOTE_SEPARATOR = re.compile(r"(?:^|\s)--\s")

# Blocks this test executes but does not compare, keyed by the first line of the
# block's code, mapped to the reason. Empty, and that is a measurement: every
# python block of the guide quotes output, and every block seeds the sampler it
# runs, so every quoted value is reproducible. A block that quotes no output fails
# this test by name instead of being skipped, and this mapping is where an
# exemption is recorded rather than there.
UNCHECKED_BLOCKS: dict[str, str] = {}


def _execute_first_python_block(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    match = PYTHON_FENCE.search(path.read_text(encoding="utf-8"))

    assert match is not None
    namespace: dict[str, object] = {"__name__": "__documentation_example__"}
    exec(
        compile(match.group(1), f"{relative_path}:first-python-block", "exec"),
        namespace,
    )
    return namespace


@dataclass(frozen=True)
class _QuotedRun:
    """The output the guide quotes for one print statement of a block.

    Attributes:
        print_line: The document line of the ``print`` statement, 1-based.
        tokens: The quoted values split into whitespace-separated tokens, in order,
            each with the document line of the comment it is quoted on.
    """

    print_line: int
    tokens: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _GuideBlock:
    """One fenced python block of the guide, with the output it quotes.

    Attributes:
        index: The block's position in the document, from zero.
        first_line: The document line of the block's first code line, 1-based.
        source: The block's code.
        runs: One entry per ``print`` statement of the block, in source order.
    """

    index: int
    first_line: int
    source: str
    runs: tuple[_QuotedRun, ...]


def _document_line(text: str, offset: int) -> int:
    """Return the 1-based document line holding the character at ``offset``."""
    return text.count("\n", 0, offset) + 1


def _python_blocks(text: str) -> tuple[_GuideBlock, ...]:
    """Return every fenced python block of ``text``, in document order."""
    blocks: list[_GuideBlock] = []
    for index, match in enumerate(PYTHON_FENCE.finditer(text)):
        source = match.group(1)
        first_line = _document_line(text, match.start(1))
        blocks.append(
            _GuideBlock(
                index=index,
                first_line=first_line,
                source=source,
                runs=_quoted_runs(source, first_line),
            )
        )
    return tuple(blocks)


def _quoted_runs(source: str, first_line: int) -> tuple[_QuotedRun, ...]:
    """Return the output each ``print`` statement of ``source`` quotes.

    A comment line is read as quoted output only when it directly follows a print
    statement, with no blank line in between. The guide's explanatory comments are
    excluded by that rule alone: each is written before the statement it explains,
    or separated from the previous print by a blank line. Within a quoted line, the
    value before the ``--`` separator is the quoted value and the prose after it is
    not; a line with no separator at all quotes the whole comment as one output
    line; and a line that only continues the prose of the line above it carries no
    value, which is how the guide wraps a long value's prose onto its own comment
    line.

    Args:
        source: One fenced python block's code.
        first_line: The document line ``source`` starts on, 1-based.

    Returns:
        One entry per print statement, in source order.
    """
    lines = source.split("\n")
    statements = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PRINT_NAME
    ]
    runs: list[_QuotedRun] = []
    for node in sorted(statements, key=lambda item: (item.lineno, item.col_offset)):
        values: list[tuple[str, int]] = []
        prose_is_open = False
        cursor = node.end_lineno
        while cursor < len(lines) and lines[cursor].strip().startswith("#"):
            comment = lines[cursor].strip().removeprefix("#").strip()
            separator = QUOTE_SEPARATOR.search(comment)
            if separator is not None:
                value = comment[: separator.start()].strip()
                prose_is_open = True
            elif prose_is_open:
                value = ""
            else:
                value = comment
            if value:
                values.append((value, first_line + cursor))
            cursor += 1
        runs.append(
            _QuotedRun(
                print_line=first_line + node.lineno - 1,
                tokens=tuple(
                    (token, line) for value, line in values for token in value.split()
                ),
            )
        )
    return tuple(runs)


def _block_key(block: _GuideBlock) -> str:
    """Return the block's identity in ``UNCHECKED_BLOCKS``: its first code line."""
    return next(line.strip() for line in block.source.split("\n") if line.strip())


def _quoted_tokens(block: _GuideBlock) -> tuple[tuple[str, int], ...]:
    """Return every token ``block`` quotes, with the document line quoting it."""
    return tuple(token for run in block.runs for token in run.tokens)


def _printed_tokens(output: str) -> tuple[tuple[str, int], ...]:
    """Return every whitespace-separated token ``output`` printed, with its line."""
    return tuple(
        (token, line_number)
        for line_number, line in enumerate(output.splitlines(), start=1)
        for token in line.split()
    )


def _execute_block(block: _GuideBlock) -> str:
    """Execute one guide block and return what it printed."""
    namespace: dict[str, object] = {"__name__": "__documentation_example__"}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(
            compile(
                block.source,
                f"{ALGORITHMS_GUIDE}:block-{block.index}",
                "exec",
            ),
            namespace,
        )
    return captured.getvalue()


def _comparison_failure(block: _GuideBlock, output: str) -> str | None:
    """Return how ``output`` differs from the values ``block`` quotes, or ``None``.

    The comparison is on whitespace-separated tokens in order, not on lines: the
    guide wraps a long output across comment lines, and a token that moved between
    two lines of the same block is not what this test is for. A changed token, an
    added one and a missing one are all reported, each with the guide line and the
    output line that disagree.
    """
    quoted = _quoted_tokens(block)
    printed = _printed_tokens(output)
    for position, (expected, actual) in enumerate(
        zip(quoted, printed, strict=False), start=1
    ):
        if expected[0] != actual[0]:
            return (
                f"value {position} of the block: the guide quotes {expected[0]!r} at "
                f"line {expected[1]}, and the block printed {actual[0]!r} at output "
                f"line {actual[1]}"
            )
    if len(quoted) > len(printed):
        token, line = quoted[len(printed)]
        return (
            f"the guide quotes {len(quoted)} values and the block printed "
            f"{len(printed)}; the first {len(printed)} agree, and then the guide "
            f"quotes {token!r} at line {line}, which the block did not print"
        )
    if len(printed) > len(quoted):
        token, line = printed[len(quoted)]
        return (
            f"the guide quotes {len(quoted)} values and the block printed "
            f"{len(printed)}; the first {len(quoted)} agree, and then the block "
            f"printed {token!r} at output line {line}, which the guide does not quote"
        )
    return None


@pytest.mark.integration
def test_algorithms_guide_examples_execute_and_match_their_quotes() -> None:
    """Execute every python block of the algorithms guide and check its quotes.

    The convention this test implements, stated once, because it is the whole of
    what makes a mismatch here mean anything:

    - The guide quotes a print statement's output in the comment lines directly
      below it, as ``# <value>  -- <prose>``. The compared part of such a line is
      the value **before** the ``--``; the prose after it is not compared, and a
      comment line that only continues that prose carries no value at all.
    - A comment line with no ``--`` is quoted output in full, which is how the
      guide's assignment transcripts and its refusal message are quoted.
    - A comment is read as quoted output only when it directly follows a print
      statement with no blank line between. The guide's explanatory comments are
      excluded by that rule and by nothing else.
    - Values are compared in order, as text, so a changed digit fails and a value
      that moved between two lines of the same block does not. A block that prints
      more or fewer values than it quotes fails either way.
    - Every block is executed twice and the two outputs must be identical: the
      comparison assumes the quoted values are reproducible, and every block of
      this guide passes an explicit seed to the sampler it runs.

    Nothing here pins the number of blocks or the sections they sit in. The blocks
    are whatever the file declares when this runs, so a section added to the guide
    is executed and compared by this test as soon as it lands. A block this test
    cannot compare -- one quoting no output, or one whose output is not
    reproducible -- fails it by name, and the place to record an exemption is
    ``UNCHECKED_BLOCKS`` above, which is empty.
    """
    text = (ROOT / ALGORITHMS_GUIDE).read_text(encoding="utf-8")
    blocks = _python_blocks(text)
    assert blocks, f"{ALGORITHMS_GUIDE} declares no python block to execute"

    failures: list[str] = []
    for block in blocks:
        name = f"{ALGORITHMS_GUIDE} block {block.index} (first line {block.first_line})"
        exempt = _block_key(block) in UNCHECKED_BLOCKS
        if not _quoted_tokens(block) and not exempt:
            failures.append(
                f"{name}: the block quotes no output for this test to compare; quote "
                "it, or record why it cannot be compared in UNCHECKED_BLOCKS"
            )
            continue
        first = _execute_block(block)
        if first != _execute_block(block):
            failures.append(
                f"{name}: two runs of the block printed different output, so its "
                "quotes cannot be compared; record why in UNCHECKED_BLOCKS"
            )
            continue
        if exempt:
            continue
        mismatch = _comparison_failure(block, first)
        if mismatch is not None:
            failures.append(f"{name}: {mismatch}")

    assert not failures, "\n".join(failures)


@pytest.mark.integration
def test_runtime_result_contract_entry_example_executes() -> None:
    namespace = _execute_first_python_block("docs/reference/RUNTIME_RESULT_CONTRACT.md")

    assert namespace["result"].plan is not None


@pytest.mark.integration
def test_hybrid_runtime_entry_example_executes() -> None:
    pytest.importorskip("jax", reason="JAX is an optional backend")
    namespace = _execute_first_python_block(
        "docs/architecture/HYBRID_RUNTIME_ARCHITECTURE.md"
    )

    assert namespace["loss"].grad_fn is not None
