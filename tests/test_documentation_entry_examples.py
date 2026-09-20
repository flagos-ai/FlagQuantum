from __future__ import annotations

import ast
import builtins
import contextlib
import io
import re
import sys
from collections.abc import Callable
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
    """One ``print`` statement of a block, with the output the guide quotes for it.

    Attributes:
        print_line: The document line the statement starts on, 1-based. This is the
            statement's identity in a failure message, and the start of the span a
            call is attributed to.
        last_line: The document line the statement ends on, 1-based. A statement may
            span several lines, and the interpreter reports a call on whichever of
            them it emitted the call from, so attribution uses the whole span.
        tokens: The quoted values split into whitespace-separated tokens, in order,
            each with the document line of the comment it is quoted on.
    """

    print_line: int
    last_line: int
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
                last_line=first_line + node.end_lineno - 1,
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


def _recording_print(records: list[tuple[int, str]]) -> Callable[..., None]:
    """Return a ``print`` that records the line it was called from and its text.

    The line is taken from the calling frame, so a call is attributed to the
    statement that made it rather than to the block: the injected name is the only
    way a block's own code reaches this function, so the caller's line is always a
    line of the block. The call is then passed through to the real ``print``, which
    keeps the block's output its own -- the reproducibility check in the test below
    still compares what the block printed.

    Args:
        records: The list each call appends ``(call line, printed text)`` to.

    Returns:
        A callable with ``print``'s own signature.
    """

    def recorded(*values: object, **keywords: object) -> None:
        separator = keywords.get("sep", " ")
        text = str(separator).join(str(value) for value in values)
        records.append((sys._getframe(1).f_lineno, text))
        builtins.print(*values, **keywords)

    return recorded


def _execute_block(block: _GuideBlock) -> tuple[tuple[int, str], ...]:
    """Execute one guide block and return one entry per ``print`` call it made.

    Each entry is ``(line, text)``: the **document** line the call was made from,
    and the text that call printed. A statement inside a loop contributes one entry
    per iteration, all carrying the statement's own line, which is what makes the
    per-statement grouping below possible. The line the interpreter reports is
    relative to the compiled block, so it is shifted by the block's own first line,
    the same shift ``_quoted_runs`` applies to the statement lines it records.

    Args:
        block: The block to execute.

    Returns:
        The calls the block made, in execution order.
    """
    records: list[tuple[int, str]] = []
    namespace: dict[str, object] = {
        "__name__": "__documentation_example__",
        PRINT_NAME: _recording_print(records),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(
            compile(
                block.source,
                f"{ALGORITHMS_GUIDE}:block-{block.index}",
                "exec",
            ),
            namespace,
        )
    return tuple((block.first_line + line - 1, text) for line, text in records)


def _statement_index(block: _GuideBlock, line: int) -> int | None:
    """Return the index of the statement whose source span holds ``line``.

    A statement's span is its first line through its last, because a call may be
    reported on any line of a multi-line ``print``. A line inside two spans would
    mean a print inside a print, which a block does not have; the narrowest match
    is taken rather than the first, so nesting would still pick the inner one.
    """
    matches = [
        index
        for index, run in enumerate(block.runs)
        if run.print_line <= line <= run.last_line
    ]
    if not matches:
        return None
    return min(matches, key=lambda index: block.runs[index].last_line)


def _per_statement_tokens(
    block: _GuideBlock, records: tuple[tuple[int, str], ...]
) -> tuple[list[list[tuple[str, int]]], list[int]]:
    """Group the recorded calls by the statement each was made from.

    Args:
        block: The block the records came from.
        records: The calls the block made.

    Returns:
        ``(tokens, unattributed)``: one token list per statement, each token carried
        with the block line of the call that printed it, and the lines of the calls
        that belong to no statement of the block.
    """
    tokens: list[list[tuple[str, int]]] = [[] for _ in block.runs]
    unattributed: list[int] = []
    for line, text in records:
        index = _statement_index(block, line)
        if index is None:
            unattributed.append(line)
            continue
        tokens[index].extend((token, line) for token in text.split())
    return tokens, unattributed


def _difference(
    where: str,
    quoted: tuple[tuple[str, int], ...],
    printed: list[tuple[str, int]],
) -> str | None:
    """Return how ``printed`` differs from ``quoted`` for one statement, or ``None``.

    The comparison is on whitespace-separated tokens in order, not on lines: the
    guide wraps a long transcript across comment lines, and the transcript of a
    statement that prints inside a loop is several lines of its own. A changed
    token, an added one and a missing one are all reported, each with the guide line
    and the block line that disagree.
    """
    for position, (expected, actual) in enumerate(
        zip(quoted, printed, strict=False), start=1
    ):
        if expected[0] != actual[0]:
            return (
                f"{where}: value {position} of the statement -- the guide quotes "
                f"{expected[0]!r} at guide line {expected[1]}, and the statement "
                f"printed {actual[0]!r} from its call at line {actual[1]}"
            )
    if len(quoted) > len(printed):
        token, line = quoted[len(printed)]
        return (
            f"{where}: the guide quotes {len(quoted)} values and the statement "
            f"printed {len(printed)}; the first {len(printed)} agree, and then the "
            f"guide quotes {token!r} at guide line {line}, which the statement did "
            "not print"
        )
    if len(printed) > len(quoted):
        token, line = printed[len(quoted)]
        return (
            f"{where}: the guide quotes {len(quoted)} values and the statement "
            f"printed {len(printed)}; the first {len(quoted)} agree, and then the "
            f"statement printed {token!r} from its call at line {line}, which the "
            "guide does not quote"
        )
    return None


def _block_failures(
    name: str, block: _GuideBlock, records: tuple[tuple[int, str], ...]
) -> list[str]:
    """Return everything wrong with one block's quotes, given what it printed."""
    tokens, unattributed = _per_statement_tokens(block, records)
    failures = [
        f"{name}: the call at line {line} belongs to no print statement of the "
        "block, so its output cannot be attributed to one"
        for line in unattributed
    ]
    for index, run in enumerate(block.runs):
        where = f"{name}, the print statement at line {run.print_line}"
        difference = _difference(where, run.tokens, tokens[index])
        if difference is not None:
            failures.append(difference)
    return failures


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
    - **Each print statement is checked on its own.** The block is executed with a
      recording ``print`` in its namespace, so every call carries the block line it
      was made from, and a call's text is compared against the transcript of *the
      statement that made it*. A block is therefore a set of statements whose
      quotes are checked one by one, not one flat stream of values: a value quoted
      against the statement next to the one that printed it fails, naming the
      statement, the guide line of the quote and the block line of the call. A
      statement that prints inside a loop contributes one call per iteration, all
      grouped under that one statement.
    - Within a statement the values are compared as text, in order, so a changed
      digit, a changed format and a value printed but not quoted all fail, and a
      value quoted but not printed fails.
    - Every block is executed twice and the two runs must have made the same calls
      with the same text: the comparison assumes the quoted values are reproducible,
      and every block of this guide passes an explicit seed to the sampler it runs.

    **What this test does not check, stated rather than left to be discovered.** It
    checks values, not prose, and it checks them as one sequence per statement:

    - Anything after a ``--``, and any comment line that is prose continuation by
      position, is not compared at all. A false sentence in the guide's prose keeps
      this test green while the values beside it are right; a quote block moved
      between two statements that print the same text is invisible for the same
      reason, because the values move with it and only the prose is left behind.
    - Within one statement's transcript the compared thing is the sequence of tokens
      and not the lines: the same values may be quoted on one comment line or split
      across several, and a value may be moved to a neighbouring comment line of
      that statement without failing, as long as the sequence does not change. That
      is what lets a statement that prints inside a loop quote one line per
      iteration. Swapping two values of the sequence, or moving one past another,
      fails, and names the position.
    - A statement that prints nothing but whitespace and quotes nothing is vacuous,
      and is checked as such.

    Everything else a quoted value can do fails: a changed value, a deleted quote, a
    quote attached to a statement that did not print it, and a statement printing a
    value the guide does not quote for it.

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
        records = _execute_block(block)
        if records != _execute_block(block):
            failures.append(
                f"{name}: two runs of the block printed different output, so its "
                "quotes cannot be compared; record why in UNCHECKED_BLOCKS"
            )
            continue
        if exempt:
            continue
        failures.extend(_block_failures(name, block, records))

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
