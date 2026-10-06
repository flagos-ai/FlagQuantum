"""Execute the composition guide's python blocks and check the output they quote.

The convention this test applies -- a print statement's output quoted in the comment
lines directly below it, compared token by token and statement by statement, with
every block executed twice -- is documented in full, and with its own list of what it
cannot catch, in
:mod:`tests.test_documentation_entry_examples`. That module owns the machinery and
this test reuses it rather than restating it, so the two guides are checked by one
convention instead of two that can drift apart.

What this adds is not a second convention but a second file. The composition guide
quotes the output of thirteen blocks, and a quoted transcript that nothing executes is
a transcript that can silently stop being true; ``docs/guides/ALGORITHMS.md`` is
checked by the module above for exactly that reason, and this guide is held to the same
rule.

Two properties of *this* guide are relied on by the check and are worth stating here,
because they are properties of the guide rather than of the convention:

- Every block is self-contained: it imports what it uses and builds its own circuits,
  so no block depends on a name an earlier block left behind. The blocks are executed
  in their own namespaces, one per block, and a block that needed an earlier one would
  fail rather than pass quietly.
- Every block is deterministic. The guide prints qubit labels, gate names, planner
  modes and refusal messages, none of which depend on the installed BLAS or on a
  random seed; the one table it states without a block is the planner reading, which
  is quoted as prose beside the block that measures it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_documentation_entry_examples import (
    _block_failures,
    _block_key,
    _execute_block,
    _python_blocks,
    _quoted_tokens,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
GUIDE = "docs/guides/CIRCUIT_COMPOSITION.md"

# Blocks this test executes but does not compare, keyed by the block's first code
# line. Empty, and that is a measurement rather than a placeholder: every block of
# this guide quotes output, and every block is deterministic. A block that quotes
# nothing fails this test by name, and this mapping is where an exemption would be
# recorded instead of in a skip.
UNCHECKED_BLOCKS: dict[str, str] = {}


def test_composition_guide_blocks_execute_and_match_their_quotes() -> None:
    """Run every python block of the composition guide and check its quotes."""
    text = (ROOT / GUIDE).read_text(encoding="utf-8")
    blocks = _python_blocks(text)

    assert blocks, f"{GUIDE} declares no python block to execute"

    failures: list[str] = []
    for block in blocks:
        name = f"{GUIDE} block {block.index} (first line {block.first_line})"
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


def test_the_guide_is_reachable_from_the_guides_index() -> None:
    """The guides index links the guide, so a reader can find it.

    A guide that exists and is linked from nowhere is a defect this repository has
    recorded before rather than a matter of taste: ``docs/guides/README.md`` is the
    hand-maintained index of the directory, and every other guide is in it.
    """
    index = (ROOT / "docs/guides/README.md").read_text(encoding="utf-8")

    assert "](CIRCUIT_COMPOSITION.md)" in index


def test_the_contract_the_guide_names_is_the_one_that_exists() -> None:
    """The contract path and the gate command the guide names both resolve.

    The guide's closing section sends a reader to the contract and to the command
    that checks it. A guide is allowed to be out of date about prose; it is not
    allowed to name a file or a command that is not there, because the reader's next
    step is to run it.
    """
    text = (ROOT / GUIDE).read_text(encoding="utf-8")

    assert "contracts/circuit-composition-contract.toml" in text
    assert (ROOT / "contracts/circuit-composition-contract.toml").is_file()
    assert "tools/check_circuit_composition_contract.py" in text
    assert (ROOT / "tools/check_circuit_composition_contract.py").is_file()
