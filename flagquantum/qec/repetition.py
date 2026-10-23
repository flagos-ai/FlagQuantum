"""Historical repetition imports; Core owns these compatibility aliases."""

from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass

from flagquantum.compiler._hybrid import INDEX as INDEX
from flagquantum.compiler._hybrid import capture_source as capture_source
from flagquantum.compiler._hybrid import lower_dynamic_program as lower_dynamic_program
from flagquantum.noise import NoiseModel as NoiseModel
from flagquantum.qec.decoders import Decoder as Decoder
from flagquantum.qec.decoders import RepetitionLookupDecoder as RepetitionLookupDecoder
from flagquantum.qec.decoders import (
    RepetitionStreamingLookupDecoder as RepetitionStreamingLookupDecoder,
)
from flagquantum.qec.decoders import (
    RepetitionTemporalDecoder as RepetitionTemporalDecoder,
)
from flagquantum.qec.decoders import StreamingDecoder as StreamingDecoder
from flagquantum.qec.decoders import _repetition_wire as _repetition_wire
from flagquantum.qec.types import Correction as Correction
from flagquantum.qec.types import DecodeResult as DecodeResult
from flagquantum.qec.types import DetectionEvent as DetectionEvent
from flagquantum.qec.types import ErrorSchedule as ErrorSchedule
from flagquantum.qec.types import PauliFrame as PauliFrame
from flagquantum.qec.types import RepetitionMemoryResult as RepetitionMemoryResult
from flagquantum.qec.types import RepetitionMemoryShot as RepetitionMemoryShot
from flagquantum.qec.types import SyndromeRound as SyndromeRound
from flagquantum.runtime.dynamic._feedback import (
    DynamicFeedbackAction as DynamicFeedbackAction,
)
from flagquantum.runtime.dynamic._feedback import (
    DynamicFeedbackObservation as DynamicFeedbackObservation,
)
from flagquantum.runtime.dynamic._feedback import (
    DynamicFeedbackPlan as DynamicFeedbackPlan,
)
from flagquantum.runtime.dynamic._feedback import (
    DynamicFeedbackPoint as DynamicFeedbackPoint,
)
from flagquantum.runtime.dynamic.hybrid_session import (
    execute_hybrid_dynamic_session as execute_hybrid_dynamic_session,
)

from .execution.repetition import _FEEDBACK_MODES as _FEEDBACK_MODES
from .execution.repetition import __all__ as __all__
from .execution.repetition import _adjust_for_frame as _adjust_for_frame
from .execution.repetition import _feedback_plan as _feedback_plan
from .execution.repetition import _lookup_correction as _lookup_correction
from .execution.repetition import _memory_source as _memory_source
from .execution.repetition import (
    _RepetitionFeedbackController as _RepetitionFeedbackController,
)
from .execution.repetition import _syndrome_rounds as _syndrome_rounds
from .execution.repetition import (
    run_repetition_memory_experiment as run_repetition_memory_experiment,
)
