"""Bind named clean QROAM outputs to dataset-major physical banks."""

from collections.abc import Mapping

from .partition import RegisterLayout, unpartition_register


def clean_bank_output_layout(
    targets: RegisterLayout, block_size: int, *, junk_prefix: str
) -> RegisterLayout:
    """List selected words first and the retained cells of each dataset second."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("QROAM block size must be a positive integer")
    return (
        *targets,
        *(
            (f"{junk_prefix}{name}", width * (block_size - 1))
            for name, width in targets
            if block_size > 1
        ),
    )


def bind_clean_banks(
    outputs: Mapping[str, tuple[int, ...]],
    targets: RegisterLayout,
    block_size: int,
    *,
    junk_prefix: str,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Group each selected word with its retained words in cell order."""
    unpartition_register(
        outputs, clean_bank_output_layout(targets, block_size, junk_prefix=junk_prefix)
    )
    if any(width < 1 for _, width in targets):
        raise ValueError("QROAM bank words must be nonempty")
    return tuple(
        (
            outputs[name],
            *(
                outputs[f"{junk_prefix}{name}"][cell * width : (cell + 1) * width]
                for cell in range(block_size - 1)
            ),
        )
        for name, width in targets
    )
