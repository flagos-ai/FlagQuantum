"""Historical imports for :mod:`flagquantum.qec.dem.chunks`; Core-owned compatibility boundary."""

from __future__ import annotations

from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from dataclasses import replace as replace
from typing import ClassVar as ClassVar

from flagquantum.qec.circuits.memory import MemoryCircuit as MemoryCircuit
from flagquantum.qec.dem.chunks import ChunkLayout as ChunkLayout
from flagquantum.qec.dem.chunks import DemChunk as DemChunk
from flagquantum.qec.dem.chunks import DemChunkSpec as DemChunkSpec
from flagquantum.qec.dem.chunks import DemChunksSpec as DemChunksSpec
from flagquantum.qec.dem.chunks import DemSeam as DemSeam
from flagquantum.qec.dem.chunks import PhaseId as PhaseId
from flagquantum.qec.dem.chunks import SeamId as SeamId
from flagquantum.qec.dem.chunks import __all__ as __all__
from flagquantum.qec.dem.chunks import _cut as _cut
from flagquantum.qec.dem.chunks import dem_chunk_from_spec as dem_chunk_from_spec
from flagquantum.qec.dem.chunks import dem_chunks_from_spec as dem_chunks_from_spec
from flagquantum.qec.dem.chunks import dem_close as dem_close
from flagquantum.qec.dem.chunks import dem_close_all as dem_close_all
from flagquantum.qec.dem.chunks import dem_stitch as dem_stitch
from flagquantum.qec.dem.chunks import dem_stitch_all as dem_stitch_all
from flagquantum.qec.dem.chunks import dem_stitch_merged as dem_stitch_merged
from flagquantum.qec.dem.model import DemError as DemError
from flagquantum.qec.dem.model import DemMergeRule as DemMergeRule
from flagquantum.qec.dem.model import DetectorErrorModel as DetectorErrorModel
from flagquantum.qec.dem.model import _count as _count
