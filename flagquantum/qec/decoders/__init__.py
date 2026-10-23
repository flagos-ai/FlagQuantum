"""Historical imports for :mod:`flagquantum.qec.decoders.repetition`; Core-owned compatibility boundary."""

from __future__ import annotations

from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from typing import Protocol as Protocol
from typing import runtime_checkable as runtime_checkable

from flagquantum.qec.decoders.repetition import Decoder as Decoder
from flagquantum.qec.decoders.repetition import (
    RepetitionLookupDecoder as RepetitionLookupDecoder,
)
from flagquantum.qec.decoders.repetition import (
    RepetitionStreamingLookupDecoder as RepetitionStreamingLookupDecoder,
)
from flagquantum.qec.decoders.repetition import (
    RepetitionTemporalDecoder as RepetitionTemporalDecoder,
)
from flagquantum.qec.decoders.repetition import StreamingDecoder as StreamingDecoder
from flagquantum.qec.decoders.repetition import __all__ as __all__
from flagquantum.qec.decoders.repetition import _repetition_wire as _repetition_wire
from flagquantum.qec.decoders.repetition import (
    _validate_repetition_history as _validate_repetition_history,
)
from flagquantum.qec.types import Correction as Correction
from flagquantum.qec.types import DecodeResult as DecodeResult
from flagquantum.qec.types import PauliFrame as PauliFrame
from flagquantum.qec.types import SyndromeRound as SyndromeRound
