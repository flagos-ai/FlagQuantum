"""Named output bindings preserve cell order and validate wire ownership."""

import pytest

from flagquantum.compiler.registers.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
)

pytestmark = pytest.mark.unit


def test_bind_selected_words_and_dataset_major_junk():
    targets = (("flag", 1), ("word", 2))
    assert clean_bank_output_layout(targets, 1, junk_prefix="junk_") == targets
    outputs = {"flag": (1,), "word": (6, 4)}
    assert bind_clean_banks(outputs, targets, 1, junk_prefix="junk_") == (
        ((1,),),
        ((6, 4),),
    )
    assert clean_bank_output_layout(targets, 3, junk_prefix="junk_") == (
        *targets,
        ("junk_flag", 2),
        ("junk_word", 4),
    )
    outputs.update(junk_flag=(7, 0), junk_word=(9, 2, 5, 3))
    assert bind_clean_banks(outputs, targets, 3, junk_prefix="junk_") == (
        ((1,), (7,), (0,)),
        ((6, 4), (9, 2), (5, 3)),
    )
    outputs["junk_word"] = (9, 2, 6, 3)
    with pytest.raises(ValueError, match="distinct"):
        bind_clean_banks(outputs, targets, 3, junk_prefix="junk_")
