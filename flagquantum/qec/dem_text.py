"""Historical imports for :mod:`flagquantum.qec.dem.text`; Core-owned compatibility boundary."""

from __future__ import annotations

from collections.abc import Iterator as Iterator
from typing import TypeAlias as TypeAlias

from flagquantum.qec.dem.text import _BLOCK_CLOSE as _BLOCK_CLOSE
from flagquantum.qec.dem.text import _BLOCK_OPEN as _BLOCK_OPEN
from flagquantum.qec.dem.text import _COMMENT as _COMMENT
from flagquantum.qec.dem.text import _DETECTOR_PREFIX as _DETECTOR_PREFIX
from flagquantum.qec.dem.text import _OBSERVABLE_PREFIX as _OBSERVABLE_PREFIX
from flagquantum.qec.dem.text import _REPEAT as _REPEAT
from flagquantum.qec.dem.text import _SEPARATOR as _SEPARATOR
from flagquantum.qec.dem.text import __annotations__ as __annotations__
from flagquantum.qec.dem.text import _declared_count as _declared_count
from flagquantum.qec.dem.text import _Entry as _Entry
from flagquantum.qec.dem.text import _instructions as _instructions
from flagquantum.qec.dem.text import _NestedInstructions as _NestedInstructions
from flagquantum.qec.dem.text import _parse_declaration_line as _parse_declaration_line
from flagquantum.qec.dem.text import _parse_error_line as _parse_error_line
from flagquantum.qec.dem.text import _parse_shift_detectors as _parse_shift_detectors
from flagquantum.qec.dem.text import _parse_stim_text as _parse_stim_text
from flagquantum.qec.dem.text import _repeat_count as _repeat_count
from flagquantum.qec.dem.text import _structure as _structure
from flagquantum.qec.dem.text import _target_index as _target_index
from flagquantum.qec.dem.text import _toggle as _toggle
from flagquantum.qec.dem.text import _unroll_stim_text as _unroll_stim_text
from flagquantum.qec.dem.text import _UnrollFrame as _UnrollFrame
