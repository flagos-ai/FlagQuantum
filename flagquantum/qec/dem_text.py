"""The stim detector error model text format, read without a stim dependency.

Stim prints a detector error model as a small instruction stream:

* ``error(<p>) <targets>`` states one mechanism at probability ``p`` and names
  the detectors and logical observables it flips;
* ``detector[(<coordinates>)] D<i>`` and ``logical_observable L<i>`` declare the
  shape indices the error targets refer to;
* ``shift_detectors[(<coordinates>)] <n>`` makes every detector index on the
  lines that follow relative to ``n`` more than the indices before it, so stim
  can address a repeated round without knowing its absolute offset;
* ``repeat <n> { ... }`` states a block of instructions to interpret ``n`` times;
* ``#`` starts a comment, which runs to the end of its line and is not part of
  the model.

This module reads that format and states what it holds as plain fields. It
depends on nothing in :mod:`flagquantum.qec.dem`, so the format can be read
without the model, and the model names the fields a reading produces. ``stim``
itself is never imported: the format is a text format, and a caller who wants
stim's own reader has stim.

A block is stated by interpreting it, which is what makes the model it produces
the model the text states: the shape, the mechanisms and the accumulated
``shift_detectors`` offsets of a block are the ones its instructions produce in
sequence, so ``repeat`` is read by running the block's instructions in order
rather than by a second expansion with its own arithmetic. That is also the one
place this reader costs more than stim: stim keeps the block as a block and
expands it on demand, and a reading here is the expansion itself.

Every construct outside that grammar is refused with the instruction named, so a
text that carries information this layer cannot state fails closed instead of
being read as a shorter model. The refusals are a malformed line, a declaration
that skips an index, a closing brace with no block open, a block that never
closes, a ``repeat`` whose count is missing or is not a run of ASCII digits, and
an error mechanism that flips nothing.

The shape is read from the declarations: the detector count is the declared
detectors and the observable count is the declared observables, or -- where stim
declares none, because a mechanism already references them -- the largest index an
error mechanism names. A text that states its detector count only through its
error targets is therefore read as a model with no detector, which the model
refuses; stim infers that count instead, so ``error(0.1) D0``, the text stim writes
for a flat one-detector circuit, is a text this reader refuses by design rather
than a construct it cannot reach.
"""

from __future__ import annotations

from collections.abc import Iterator

_DETECTOR_PREFIX = "D"
_OBSERVABLE_PREFIX = "L"
_SEPARATOR = "^"
_COMMENT = "#"
_BLOCK_OPEN = "{"
_BLOCK_CLOSE = "}"
_REPEAT = "repeat"

# One entry of a parsed error line: the probability it states and the detectors
# and observables a shot of it flips, each sorted, deduplicated and relative to
# the shifts in force at that line.
_Entry = tuple[float, tuple[int, ...], tuple[int, ...]]


def _target_index(token: str, prefix: str) -> int | None:
    """Return the index a stim target names, or ``None`` if it names nothing.

    A target is a one-letter prefix followed by ASCII digits, so a token such
    as ``X0`` or ``D`` is a malformed target rather than a crash inside
    ``int``, which accepts the digits of other scripts.
    """

    digits = token[len(prefix) :]
    if token.startswith(prefix) and digits.isascii() and digits.isdigit():
        return int(digits)
    return None


def _parse_shift_detectors(line: str) -> int:
    """Return the detector-index shift a ``shift_detectors`` line states.

    The instruction takes optional coordinates in parentheses and exactly one
    target, the shift itself. Coordinates carry geometry the model does not
    represent, so they are accepted and discarded for the same reason a
    ``detector`` declaration's coordinates are: the index is the target.

    Stim reaches the absolute detector index by adding the shifts accumulated so
    far, so the caller accumulates this value rather than replacing one.
    """

    remainder = line[len("shift_detectors") :]
    if remainder.startswith("("):
        closing = remainder.find(")")
        if closing < 0:
            raise ValueError("shift_detectors coordinates must close in parentheses")
        remainder = remainder[closing + 1 :]
    tokens = remainder.split()
    if len(tokens) != 1:
        raise ValueError(
            "a shift_detectors instruction must state exactly one "
            "detector-index shift"
        )
    shift = tokens[0]
    # Stim accepts an unsigned integer here and nothing else, so a signed or
    # non-ASCII token is malformed rather than a negative shift in disguise.
    if not shift.isascii() or not shift.isdigit():
        raise ValueError(
            "a shift_detectors instruction must state a non-negative integer "
            f"shift, not {shift!r}"
        )
    return int(shift)


def _toggle(targets: set[int], index: int) -> None:
    """Flip one target's membership, which is how a repeated target cancels."""

    if index in targets:
        targets.remove(index)
    else:
        targets.add(index)


def _parse_error_line(
    line: str, *, detector_shift: int, decomposed: bool = False
) -> tuple[_Entry, ...]:
    """Parse one ``error(<p>) <targets...>`` line into its mechanisms.

    A detector target is relative to the shifts in force at that line, which is
    what makes a ``shift_detectors`` instruction meaningful; an observable
    target is absolute, because the format has no observable shift.

    Stim's ``^`` separators mark how a composite mechanism decomposes into
    simpler ones; they partition the targets and do not change which detectors
    and observables a shot of this mechanism flips. Read one way, the line is a
    single mechanism whose signature is the symmetric difference of all its
    targets, and the groups are not retained. Read the other way, each group
    becomes a mechanism of its own at the parent probability, which is the
    decomposition suggestion the separators carry and is a lossy reading of the
    line rather than an equivalent one: two components that can each fire
    independently at probability ``p`` do not reproduce a single mechanism at
    probability ``p``. ``decomposed`` selects between the two, and the combined
    reading is the default because it is the one that preserves the model.

    A target that appears twice cancels, within a group and across groups alike,
    which is the symmetric difference the combined reading states. Two
    components with no targets at all are also possible -- ``error(0.1) D0 ^
    D0`` states one twice -- and neither reading drops what it cannot state: the
    combined reading refuses a line whose whole signature is empty, and the
    decomposed reading refuses a component whose own signature is empty, because
    a mechanism that flips nothing is not a mechanism this model can hold.
    """

    body = line[len("error") :]
    if not body.startswith("("):
        raise ValueError("an error line must state its probability in parentheses")
    body = body[1:]
    closing = body.find(")")
    if closing < 0:
        raise ValueError("an error line must close its probability in parentheses")
    try:
        probability = float(body[:closing])
    except ValueError as error:
        raise ValueError("an error line must state a numeric probability") from error
    tokens = body[closing + 1 :].split()
    if not tokens:
        raise ValueError(
            "an error mechanism must flip at least one detector or observable"
        )
    groups: list[tuple[set[int], set[int]]] = []
    detectors: set[int] = set()
    observables: set[int] = set()
    group_is_empty = True
    for token in tokens:
        if token == _SEPARATOR:
            if group_is_empty:
                raise ValueError(
                    "an error separator must sit between two groups of targets"
                )
            groups.append((detectors, observables))
            detectors, observables = set(), set()
            group_is_empty = True
            continue
        group_is_empty = False
        detector = _target_index(token, _DETECTOR_PREFIX)
        if detector is not None:
            _toggle(detectors, detector + detector_shift)
            continue
        observable = _target_index(token, _OBSERVABLE_PREFIX)
        if observable is not None:
            _toggle(observables, observable)
            continue
        if _SEPARATOR in token:
            raise ValueError("an error separator must be separated by spacing")
        raise ValueError(f"error targets must be D or L indices, not {token!r}")
    if group_is_empty:
        raise ValueError("an error separator must sit between two groups of targets")
    groups.append((detectors, observables))

    if decomposed and len(groups) > 1:
        # Upstream's `use_decomp_suggestions`: one column per component, each
        # inheriting the parent instruction's probability. A component that
        # flips nothing has no column to become, and dropping it silently would
        # report a model with a mechanism removed, so the line is refused and
        # the combined reading is named as the route that states it.
        entries: list[_Entry] = []
        for group_detectors, group_observables in groups:
            if not group_detectors and not group_observables:
                raise ValueError(
                    "a decomposition component that flips nothing cannot be a "
                    "mechanism of its own; read the line without "
                    "use_decomp_suggestions to state its combined signature"
                )
            entries.append(
                (
                    probability,
                    tuple(sorted(group_detectors)),
                    tuple(sorted(group_observables)),
                )
            )
        return tuple(entries)

    combined_detectors: set[int] = set()
    combined_observables: set[int] = set()
    for group_detectors, group_observables in groups:
        combined_detectors ^= group_detectors
        combined_observables ^= group_observables
    # The record this becomes re-checks the same rule; refusing here as well
    # fails at the parse site, where the offending line is still in hand.
    if not combined_detectors and not combined_observables:
        raise ValueError(
            "an error mechanism must flip at least one detector or observable"
        )
    return (
        (
            probability,
            tuple(sorted(combined_detectors)),
            tuple(sorted(combined_observables)),
        ),
    )


def _parse_declaration_line(
    line: str,
    *,
    instruction: str,
    prefix: str,
    coordinates: bool,
    detector_shift: int,
) -> int:
    """Parse one ``detector [<coordinates>] D<i>`` declaration into ``i``.

    Coordinates carry geometry the model does not represent, so a detector
    declaration accepts them and discards them: the declared index is the
    target, never the coordinate. A logical-observable declaration takes no
    coordinates, which is the form the format itself accepts. A detector index
    is relative to the shifts in force at that line.
    """

    remainder = line[len(instruction) :]
    if coordinates and remainder.startswith("("):
        closing = remainder.find(")")
        if closing < 0:
            raise ValueError(f"{instruction} coordinates must close in parentheses")
        remainder = remainder[closing + 1 :]
    tokens = remainder.split()
    if len(tokens) != 1:
        raise ValueError(
            f"a {instruction} declaration must name exactly one {prefix} index"
        )
    index = _target_index(tokens[0], prefix)
    if index is None:
        raise ValueError(
            f"a {instruction} declaration must name a {prefix} index, "
            f"not {tokens[0]!r}"
        )
    return index + (detector_shift if prefix == _DETECTOR_PREFIX else 0)


def _declared_count(declared: set[int], *, name: str, prefix: str) -> int:
    """Return how many indices the text declared, refusing a hole.

    The count may only come from the declarations, so a text that declares
    ``D1`` without ``D0`` is malformed rather than a one-detector model that
    happens to call its detector one.
    """

    count = len(declared)
    if declared != set(range(count)):
        raise ValueError(f"{name} declarations must be consecutive from {prefix}0")
    return count


def _instructions(text: str) -> list[str]:
    """Return the instruction lines a text holds, in the order it states them.

    A comment is cut at the ``#`` that starts it wherever it sits on the line,
    which is where stim cuts it too, so ``error(0.1) D0  # one`` states the same
    instruction as ``error(0.1) D0``. A line left with nothing on it states no
    instruction and is dropped, so blank lines and comment-only lines cost
    nothing but their own width.
    """

    instructions: list[str] = []
    for raw in text.splitlines():
        stated = raw.split(_COMMENT, 1)[0].strip()
        if stated:
            instructions.append(stated)
    return instructions


def _repeat_count(instruction: str) -> int:
    """Return the iteration count a ``repeat <n> {`` line states.

    The count is a run of ASCII digits: stim refuses a sign, so ``+2`` and
    ``-1`` are malformed rather than a positive count in disguise, and leading
    zeros are simply digits. The opening brace has to end the line, because stim
    puts the block's first instruction on the next line and reads anything after
    the brace as a target of the opening line.
    """

    head = instruction[: -len(_BLOCK_OPEN)].split()
    if len(head) != 2 or head[0] != _REPEAT:
        raise ValueError(
            "a block must open with `repeat <count> {` on one line, not "
            f"{instruction!r}"
        )
    count = head[1]
    if not count.isascii() or not count.isdigit():
        raise ValueError(
            f"a repeat block must state a non-negative integer count, not {count!r}"
        )
    return int(count)


def _structure(text: str) -> list[str | list]:
    """Return the text's instructions with its repeat blocks nested.

    A block is a ``[count, instructions]`` pair in its parent's list, so the
    nesting a text states is the nesting this returns. The parse is the whole of
    the format's structure: everything else is an instruction line, and a brace
    that does not open or close a block is refused here rather than reaching the
    instruction reader as a malformed target.
    """

    root: list[str | list] = []
    open_blocks: list[list[str | list]] = [root]
    for instruction in _instructions(text):
        if instruction == _BLOCK_CLOSE:
            if len(open_blocks) == 1:
                raise ValueError(
                    "a closing brace must close a repeat block, and no repeat "
                    "block is open"
                )
            open_blocks.pop()
        elif instruction.endswith(_BLOCK_OPEN):
            body: list[str | list] = []
            open_blocks[-1].append([_repeat_count(instruction), body])
            open_blocks.append(body)
        elif _BLOCK_OPEN in instruction or _BLOCK_CLOSE in instruction:
            raise ValueError(
                "a repeat block must open with a brace that ends its line and "
                f"close on a line of its own, not {instruction!r}"
            )
        else:
            open_blocks[-1].append(instruction)
    if len(open_blocks) != 1:
        raise ValueError("a repeat block must close with a brace on a line of its own")
    return root


def _unroll_stim_text(text: str) -> Iterator[str]:
    """Yield the instructions a text states, with every repeat block interpreted.

    The block is stated by running its instructions as many times as it says, so
    a block's declarations, its error mechanisms and its ``shift_detectors``
    offsets are the ones that block produces in sequence. An iteration count of
    zero runs the block no times, which is why a block that only shifts detectors
    contributes no shift at zero. Blocks nest, and an inner block is drained
    inside the iteration of the one that holds it.

    A reading is the expansion itself, so a block costs what its count states.
    Stim keeps the block and expands it on demand, which is the one cost
    difference between the two readers; the model either produces is the same.
    """

    # ``[instructions, index, iterations left]``. The root frame runs once, so
    # the same walk states a text with no block and one nested several deep.
    stack: list[list] = [[_structure(text), 0, 1]]
    while stack:
        frame = stack[-1]
        body: list[str | list] = frame[0]
        if frame[1] >= len(body):
            if frame[2] > 1:
                frame[2] -= 1
                frame[1] = 0
                continue
            stack.pop()
            continue
        instruction = body[frame[1]]
        frame[1] += 1
        if isinstance(instruction, list):
            count, block = instruction
            if count:
                stack.append([block, 0, count])
        else:
            yield instruction


def _parse_stim_text(
    text: str, *, use_decomp_suggestions: bool = False
) -> tuple[int, int, tuple[_Entry, ...]]:
    """Parse stim text into a model shape and its error mechanisms.

    Only the instructions this model represents are accepted; every other
    construct is refused with a stated reason, so a text that carries
    information the model cannot hold fails closed instead of losing it.
    ``use_decomp_suggestions`` decides whether a line's ``^`` groups are read as
    one mechanism or as one mechanism each, which ``_parse_error_line`` states
    in full.

    Each mechanism is returned as the fields the format states -- its
    probability, its detectors and its observables, each sorted, deduplicated
    and absolute -- so this layer depends on no record of its own.

    Returns:
        The detector count, the observable count, and one entry per mechanism.

    Raises:
        ValueError: A line is malformed, a declaration skips an index, a brace
            does not open or close a block, a repeat count is missing or not a
            run of ASCII digits, or an error mechanism flips nothing.
    """

    declared_detectors: set[int] = set()
    declared_observables: set[int] = set()
    referenced_observables: set[int] = set()
    detector_shift = 0
    entries: list[_Entry] = []
    for instruction in _unroll_stim_text(text):
        # The keyword is what precedes the first parenthesis, so the
        # coordinates of ``detector(1, 2) D0`` belong to the body.
        head = instruction.split("(", 1)[0].split()
        keyword = head[0] if head else ""
        if keyword == "error":
            parsed = _parse_error_line(
                instruction,
                detector_shift=detector_shift,
                decomposed=use_decomp_suggestions,
            )
            entries.extend(parsed)
            for _, _, observables in parsed:
                referenced_observables.update(observables)
        elif keyword == "detector":
            declared_detectors.add(
                _parse_declaration_line(
                    instruction,
                    instruction="detector",
                    prefix=_DETECTOR_PREFIX,
                    coordinates=True,
                    detector_shift=detector_shift,
                )
            )
        elif keyword == "logical_observable":
            declared_observables.add(
                _parse_declaration_line(
                    instruction,
                    instruction="logical_observable",
                    prefix=_OBSERVABLE_PREFIX,
                    coordinates=False,
                    detector_shift=detector_shift,
                )
            )
        elif keyword == "shift_detectors":
            detector_shift += _parse_shift_detectors(instruction)
        else:
            raise ValueError(f"unsupported stim instruction {instruction!r}")
    num_detectors = _declared_count(
        declared_detectors, name="detector", prefix=_DETECTOR_PREFIX
    )
    # Stim declares the observable when no error mechanism references it and
    # omits the declaration when one does, so at most one of these two sources
    # has anything to say. A declared shape governs; the referenced indices
    # govern only where there is no declaration to state the shape.
    num_observables = (
        _declared_count(
            declared_observables, name="logical_observable", prefix=_OBSERVABLE_PREFIX
        )
        if declared_observables
        else (max(referenced_observables) + 1 if referenced_observables else 0)
    )
    return num_detectors, num_observables, tuple(entries)
