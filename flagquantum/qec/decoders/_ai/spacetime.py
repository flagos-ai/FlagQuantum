"""Private DEM-bound detector-grid predecoder with local 3D convolutions."""

from __future__ import annotations

import math

import torch
from torch import nn

from flagquantum.qec.circuits.memory import MemoryCircuit
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.dem.model import DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise


class SpacetimePredecoder(Predecoder):
    """Encode syndrome sites and masks; pool local features onto DEM mechanisms.

    Data qubit j*side+i denotes grid coordinate (i,j). This is an explicit
    geometry assumption, not a claim that arbitrary codes have a square layout.
    Offline convolutions use both past and future; this is not streaming decode.
    """

    sites: torch.Tensor
    masks: torch.Tensor

    def __init__(self, memory: MemoryCircuit, noise: PhenomenologicalNoise) -> None:
        dem = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
        super().__init__(dem)
        side = math.isqrt(memory.code.num_data_qubits)
        if side < 2 or memory.code.data_qubits != tuple(range(side * side)):
            raise ValueError("requires row-major square-grid data wires")
        width = 2 * side - 1
        grid_shape = (memory.rounds + 1, width, width)
        checks = {check.ancilla_qubit: check for check in memory.code.checks}
        sites: list[int] = []
        kinds: list[int] = []
        for detector in memory.detectors.detectors:
            ancillas = {ref.qubit for ref in detector.parity if ref.qubit in checks}
            if len(ancillas) != 1:
                raise ValueError("detector must reference one check ancilla")
            check = checks[ancillas.pop()]
            if bool(check.stabilizer.x_qubits) == bool(check.stabilizer.z_qubits):
                raise ValueError("requires pure X or Z checks")
            support = check.stabilizer.support
            xs = [wire % side for wire in support]
            ys = [wire // side for wire in support]
            x, y = min(xs) + max(xs), min(ys) + max(ys)
            terminal = any(ref.round_index is None for ref in detector.parity)
            rounds = [
                ref.round_index
                for ref in detector.parity
                if ref.round_index is not None
            ]
            time = memory.rounds if terminal else max(rounds)
            sites.append((time * width + y) * width + x)
            kinds.append(1 if check.stabilizer.x_qubits else 2)
        self._set_geometry(grid_shape, tuple(sites), tuple(kinds))

    @classmethod
    def from_dem(
        cls,
        dem: DetectorErrorModel,
        *,
        grid_shape: tuple[int, int, int],
        sites: tuple[int, ...],
        kinds: tuple[int, ...],
    ) -> SpacetimePredecoder:
        """Use original DEM columns and a verified flattened detector-site map.

        Kinds are 1 for X and 2 for Z. Structural checks cannot authenticate
        physical coordinates or measurement-record identities; the caller must.
        Visible hyperedges remain complete, as in the explicit MLP path.
        """
        model = cls.__new__(cls)
        Predecoder.__init__(model, dem, allow_hyperedges=True)
        model._set_geometry(grid_shape, sites, kinds)
        return model

    def _set_geometry(
        self,
        grid_shape: tuple[int, int, int],
        sites: tuple[int, ...],
        kinds: tuple[int, ...],
    ) -> None:
        if (
            not isinstance(grid_shape, tuple)
            or len(grid_shape) != 3
            or any(type(size) is not int or size < 1 for size in grid_shape)
        ):
            raise ValueError("grid shape must contain three positive integers")
        volume = math.prod(grid_shape)
        if (
            not isinstance(sites, tuple)
            or len(sites) != self.h.shape[0]
            or any(type(site) is not int or not 0 <= site < volume for site in sites)
        ):
            raise ValueError("sites must map every detector into the grid")
        if len(set(sites)) != len(sites):
            raise ValueError("detector geometry has colliding sites")
        if (
            not isinstance(kinds, tuple)
            or len(kinds) != len(sites)
            or any(type(kind) is not int or kind not in (1, 2) for kind in kinds)
        ):
            raise ValueError("kinds must label every detector as X=1 or Z=2")
        self.grid_shape = grid_shape
        self.register_buffer("sites", torch.tensor(sites, dtype=torch.long))
        masks = torch.zeros(3, math.prod(self.grid_shape))
        masks[0, self.sites] = 1
        for site, kind in zip(sites, kinds, strict=True):
            masks[kind, site] = 1
        self.register_buffer("masks", masks.reshape(3, *self.grid_shape))
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.network = nn.Sequential(nn.Linear(11, 16), nn.Tanh(), nn.Linear(16, 1))

    def encode_input(self, detectors: torch.Tensor) -> torch.Tensor:
        """Return [batch, syndrome/valid/X/Z, time, row, column]."""
        if detectors.ndim != 2 or detectors.shape[1] != self.h.shape[0]:
            raise ValueError("detector batch has incompatible shape")
        if not bool(((detectors == 0) | (detectors == 1)).all()):
            raise ValueError("detectors must be binary")
        values = torch.zeros(
            (detectors.shape[0], math.prod(self.grid_shape)), device=detectors.device
        )
        values[:, self.sites] = detectors.float()
        syndrome = values.reshape(detectors.shape[0], 1, *self.grid_shape)
        return torch.cat(
            (
                syndrome,
                self.masks.unsqueeze(0).expand(detectors.shape[0], -1, -1, -1, -1),
            ),
            dim=1,
        )

    def forward(self, detectors: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(self.encode_input(detectors))
        # Gather in original detector order, then pool only incident sites onto
        # each original mechanism column. No output is inferred from truth labels.
        local = encoded.flatten(2)[:, :, self.sites].transpose(1, 2)
        degree = self.h.sum(0)
        pooled = torch.einsum("bdc,de->bec", local, self.h) / degree[None, :, None]
        active = detectors.float() @ self.h
        features = torch.cat(
            (
                pooled,
                active.unsqueeze(-1),
                degree.expand_as(active).unsqueeze(-1),
                torch.logit(self.prior).expand_as(active).unsqueeze(-1),
            ),
            dim=-1,
        )
        logits: torch.Tensor = self.network(features)
        return logits.squeeze(-1)
