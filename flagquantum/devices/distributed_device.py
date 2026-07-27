import os
from functools import partialmethod
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from ..ops import functional, matrices
from ..ops.matrices import get_global_precision
from ..utils.interchange import interchange_qubits
from ..utils.maybe_dtensor import maybe_distribute_tensor, maybe_get_dtensor_info


class DistributedQuantumDevice:
    """A distributed quantum device that contains and manages quantum state vectors.

    This device supports distributed training across multiple GPUs/nodes by sharding the
    quantum state tensor across devices. It provides automatic resharding when quantum
    gates are applied to sharded dimensions.

    Attributes:
        n_wires: Number of qubits in the quantum system.
        device_name: Name identifier for the quantum device.
        device: Classical computing device ('cpu' or 'cuda').
        bsz: Batch size for batched quantum states.
        world_sz: Number of distributed processes.
        local_rank: Local rank of the current process.
        global_rank: Global rank of the current process.
        device_mesh: Device mesh for distributed tensor operations.
        _states: The quantum state tensor (may be distributed).
        _groupings: Tensor describing qubit grouping and sharding.
        op_history: List of recorded operations if record_op is True.

    Example:
        Single GPU usage:
        >>> qdev = DistributedQuantumDevice(n_wires=4, bsz=2, device="cuda")
        >>> qdev.reset_states()
        >>> qdev.h(wires=[0])
        >>> qdev.cnot(wires=[0, 1])
        >>> states = qdev.states

        Multi-GPU distributed usage:
        >>> # Run with: torchrun --nproc_per_node=2 script.py
        >>> qdev = DistributedQuantumDevice(
        ...     n_wires=6,
        ...     bsz=4,
        ...     world_sz=2,
        ...     device="cuda"
        ... )
        >>> qdev.ry(wires=[2], theta=0.5)
        >>> qdev.cnot(wires=[1, 3])
        >>> results = qdev.states

        Custom amplitude loading:
        >>> amplitudes = torch.randn(2, 2**4, dtype=torch.complex64)
        >>> qdev.load_amplitudes(amplitudes)

        Invertible mode for backpropagation:
        >>> qdev = DistributedQuantumDevice(
        ...     n_wires=4,
        ...     invertible=True,
        ...     device="cuda"
        ... )
        >>> # Use with automatic differentiation frameworks
    """

    def __init__(
        self,
        n_wires: int,
        bsz: int = 1,
        device_name: str = "default",
        device: Union[torch.device, str] = "cuda",
        record_op: bool = False,
        world_sz: int = 1,
        shared_seed: int = 20740,
        invertible: bool = False,
        max_dtensor_dims: int = 16,
        precision: Optional[torch.dtype] = None,
    ):
        """Initializes the distributed quantum device.

        Args:
            n_wires: Number of qubits in the quantum system.
            bsz: Batch size for batched quantum states. Defaults to 1.
            device_name: Name identifier for the quantum device. Defaults to "default".
            device: Classical computing device. Can be 'cpu', 'cuda', or torch.device.
                Defaults to "cuda".
            record_op: Whether to record operations for static computation graph
                construction. Defaults to False.
            world_sz: Number of distributed processes. Set to 1 for single process.
                Defaults to 1.
            shared_seed: Random seed for reproducibility across processes.
                Defaults to 20740.
            invertible: Whether to enable invertible mode for backpropagation.
                Defaults to False.
            max_dtensor_dims: Maximum dimensions for distributed tensor. Used for
                grouping large numbers of qubits. Defaults to 16.
            precision: Optional precision parameters.

        Raises:
            KeyError: If WORLD_SIZE or RANK environment variables are not set when
                world_sz > 1.

        Example:
            Basic initialization:
            >>> qdev = DistributedQuantumDevice(n_wires=4, bsz=2)

            Distributed initialization:
            >>> qdev = DistributedQuantumDevice(
            ...     n_wires=8,
            ...     world_sz=4,
            ...     device="cuda",
            ...     invertible=True
            ... )
        """

        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = device_name + "_distributed"
        self.device = device
        self.shared_seed = shared_seed

        self.record_op = record_op
        self.op_history: list[dict[str, Any]] = []
        self.invertible = invertible

        # Bind interchange function
        self.interchange_qubits = interchange_qubits

        # set up distributed
        self.world_sz = world_sz
        if world_sz > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
        else:
            local_rank = 0
            global_rank = 0
        self.local_rank = local_rank
        self.global_rank = global_rank
        if device == "cuda":
            torch.cuda.set_device(f"cuda:{local_rank}")
        self.device_mesh = None
        self.log2_devices = (world_sz - 1).bit_length()
        if world_sz > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=global_rank,
                    world_size=world_sz,
                )
            self.device_mesh = init_device_mesh(device, (2,) * self.log2_devices)

        # First row of self._groupings indicates group number, second row indicates relative position
        self._groupings = torch.zeros((2, self.n_wires), dtype=torch.int)
        # use 1st dim for batching, last dim for real/imag
        if self.n_wires < max_dtensor_dims - 2:
            self.local_shape = (
                (1,)
                + (2,) * (self.n_wires - self.log2_devices)
                + (1,) * self.log2_devices
                + (2,)
            )
            self._groupings[0] = (
                torch.arange(self.n_wires) + 1
            )  # Each qubit is own group
            self._groupings[1] = -1
            if self.log2_devices > 0:
                self._groupings[1, -self.log2_devices :] = -2
            qubit_idx = self.n_wires - self.log2_devices
        else:
            # Number of possible dimensions for grouping equals total dimension number minus batching dim, real/imag dim, sharding dims, and two single-qubit dims
            num_grouped_dims = max_dtensor_dims - 4 - self.log2_devices
            num_grouped_qubits = self.n_wires - self.log2_devices - 2

            group_nums = [
                num_grouped_qubits // num_grouped_dims
                + int(i < num_grouped_qubits % num_grouped_dims)
                for i in range(num_grouped_dims)
            ]
            self.local_shape = (
                (1,)
                + (2,)
                + tuple([2**qubits for qubits in group_nums])
                + (2,)
                + (1,) * self.log2_devices
                + (2,)
            )

            # Arrange groupings according to grouped_dimensions
            # First and last groupings always contain only 1 qubit
            # Second row designation of -1 includes ungrouped qubit, -2 designations sharded qubit
            self._groupings[:, 0] = torch.Tensor([1, -1]).int()
            qubit_idx = 1
            for i in range(len(group_nums)):
                self._groupings[0, qubit_idx : qubit_idx + group_nums[i]] = i + 2
                if group_nums[i] == 1:
                    self._groupings[1, qubit_idx] = -1
                else:
                    self._groupings[1, qubit_idx : qubit_idx + group_nums[i]] = (
                        torch.arange(group_nums[i])
                    )
                qubit_idx += group_nums[i]
            # qubit_idx now indicates end of grouped qubits
            self._groupings[0, qubit_idx] = len(group_nums) + 2
            self._groupings[1, qubit_idx] = -1
            self._groupings[0, qubit_idx + 1 :] = (
                len(group_nums) + 3 + torch.arange(self.log2_devices)
            )
            self._groupings[1, qubit_idx + 1 :] = -2

        self.last_unsharded = (
            qubit_idx  # last unsharded group always contains one qubit
        )
        self.num_dims = max_dtensor_dims
        self._init_precision(precision)
        self.reset_states(bsz)

    def _init_precision(self, precision: Optional[torch.dtype] = None):
        """Initialize precision settings and synchronize with global PrecisionConfig.

        Args:
            precision: Optional precision specification. Accepts complex64/complex128
                    or float32/float64 (will be converted to corresponding complex type).
        """
        if precision is not None:
            # User-specified precision
            if precision in [torch.complex64, torch.complex128]:
                target_precision = precision
            elif precision in [torch.float32, torch.float64]:
                # Convert real precision to corresponding complex precision
                target_precision = (
                    torch.complex64 if precision == torch.float32 else torch.complex128
                )
            else:
                raise ValueError(f"Unsupported precision: {precision}")

        else:
            target_precision = get_global_precision()

        self._complex_dtype = target_precision
        self._real_dtype = (
            torch.float32 if target_precision == torch.complex64 else torch.float64
        )

    @property
    def real_dtype(self) -> torch.dtype:
        """Get the current real data type (float32 or float64)."""
        return self._real_dtype

    @property
    def complex_dtype(self) -> torch.dtype:
        """Get the current complex data type (complex64 or complex128)."""
        return self._complex_dtype

    @property
    def current_precision(self) -> tuple:
        """Get current precision as a tuple (real_dtype, complex_dtype).

        Returns:
            Tuple of (real_dtype, complex_dtype), e.g., (torch.float32, torch.complex64)
        """
        return self._real_dtype, self._complex_dtype

    def reset_states(
        self, bsz: Optional[int] = None, dtype: Optional[torch.dtype] = None
    ):
        """Reset quantum states to the initial |0...0⟩ state.

        This method reinitializes the quantum state tensor to the computational
        basis ground state (all qubits in |0⟩). The state tensor is reshaped
        according to the batch size and precision settings.

        Args:
            bsz: New batch size. If provided, updates the batch size and reshapes
                the local tensor accordingly. Defaults to None (keep current batch size).
            dtype: Data type for the state tensor. Accepts float32/float64 or
                complex64/complex128 (will be converted to the corresponding real type).
                If None, uses the current device precision.

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=3)
            >>> # Reset to |000⟩ with double precision
            >>> qdev.reset_states(dtype=torch.float64)
            >>> # Reset with batch size 4
            >>> qdev.reset_states(bsz=4)

        Notes:
            - The state tensor is stored as real numbers (real/imag pairs).
            - When complex types are passed via `dtype`, they are automatically
            converted to the corresponding real type (complex64 → float32,
            complex128 → float64).
            - This method automatically synchronizes the global precision setting
            if the requested precision differs from the current one.
        """
        if bsz:
            self.local_shape = tuple(
                [
                    self.local_shape[i] if i > 0 else bsz
                    for i in range(len(self.local_shape))
                ]
            )
            self.bsz = bsz

        # Resolve data type
        if dtype is None:
            dtype = self._real_dtype
        elif dtype in [torch.complex64, torch.complex128]:
            # Convert complex types to corresponding real types
            dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        target_complex = torch.complex64 if dtype == torch.float32 else torch.complex128
        if self._complex_dtype != target_complex:
            self._complex_dtype = target_complex
            self._real_dtype = dtype

        # shard along designated initial wires
        sharded_wires = (self._groupings[:, (self._groupings[1] == -2)][0]).tolist()
        self._states = torch.zeros(self.local_shape, device=self.device, dtype=dtype)

        # Initialize the state to |0...0⟩ (only rank 0 initializes, others get zeros)
        if self.global_rank == 0:
            # Set the first element (all qubits in |0⟩) to 1
            # The index (slice(None),) + (0,) * (ndim - 1) sets the first
            # position along all qubit dimensions while preserving batch dimension
            self._states[(slice(None),) + (0,) * (self._states.ndim - 1)] = 1

        # Distribute the state tensor across devices if running in distributed mode
        if self.world_sz > 1:
            placements = [Shard(i) for i in sharded_wires]
            self._states = DTensor.from_local(
                self._states, self.device_mesh, placements
            )

        # Initialize invertible dummy tensor for memory-efficient backpropagation
        if self.invertible:
            # Dummy tensor (value doesn't matter, only shape and dtype are important)
            # It acts as a placeholder for gradient flow in invertible computations
            self._invertible_dummy = torch.tensor(
                0, dtype=self._states.dtype, device=self._states.device
            )
            if self.world_sz > 1:
                # Create distributed version matching the state's sharding
                self._invertible_dummy = DTensor.from_local(
                    self._invertible_dummy.expand_as(self._states.to_local()),
                    self.device_mesh,
                    placements,
                )
            else:
                # Expand to match the full state shape (broadcasting)
                self._invertible_dummy = self._invertible_dummy.expand_as(self._states)
        else:
            self._invertible_dummy = None

    def load_amplitudes(self, amplitudes: torch.Tensor):
        """Load custom amplitudes directly into the quantum state.

        This method allows direct initialization of the quantum state with
        arbitrary amplitude vectors. The amplitudes are automatically normalized
        to unit norm.

        Args:
            amplitudes: Complex or real tensor representing quantum state amplitudes.
                Shape should be (batch_size, 2**n_wires) for complex tensors, or
                (batch_size, 2**n_wires) for real tensors (imaginary part set to zero).
                For real tensors, the imaginary part is implicitly zero.

        Raises:
            ValueError: If amplitudes shape is incompatible with n_wires.

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=2)
            >>> # Load custom amplitudes for 2 qubits (complex)
            >>> amps = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.complex64)
            >>> qdev.load_amplitudes(amps)
            >>> # Load real amplitudes (imaginary part set to zero)
            >>> amps_real = torch.randn(2, 4)
            >>> qdev.load_amplitudes(amps_real)

        Notes:
            - The device state is reset before loading new amplitudes.
            - Precision is automatically inferred from the input tensor.
            - Amplitudes are normalized so that sum(|amplitude|^2) = 1.
            - For complex inputs, the real and imaginary parts are separated
            and stored as a real tensor with shape (batch_size, ..., 2).
        """
        if amplitudes.is_complex():
            real_dtype = (
                torch.float32 if amplitudes.dtype == torch.complex64 else torch.float64
            )
            self.reset_states(bsz=amplitudes.shape[0], dtype=real_dtype)
            loading = torch.view_as_real(amplitudes).to(self.device)
        else:
            self.reset_states(bsz=amplitudes.shape[0], dtype=amplitudes.dtype)
            loading = torch.stack(
                (amplitudes, torch.zeros_like(amplitudes)), dim=-1
            ).to(self.device)

        # Normalize the amplitudes to unit norm
        norms = torch.sqrt(
            (loading**2).sum(dim=tuple(range(1, loading.ndim)))
        ).reshape((-1,) + (1,) * (loading.ndim - 1))

        # Distribute the tensor across devices if needed
        maybe_mesh, maybe_placements = maybe_get_dtensor_info(self._states)
        self._states = maybe_distribute_tensor(
            (loading / norms).reshape(self._states.shape),
            device_mesh=maybe_mesh,
            placements=maybe_placements,
        )

    @property
    def noncanonical_states(self) -> tuple[Union[DTensor, torch.Tensor], torch.Tensor]:
        """Obtains device states and groupings without canonicalization.

        The groupings are detached and cloned to prevent interference. The states are
        not cloned to avoid unnecessary computations. The expectation is that callers
        will copy the statevector rather than modifying it in-place.

        Returns:
            A tuple containing:
                - Current quantum state tensor (may be distributed)
                - Detached and cloned groupings tensor

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=3)
            >>> states, groupings = qdev.noncanonical_states
            >>> # Use states and groupings for measurement without canonicalization
        """
        return self._states, self._groupings.detach().clone()

    def canonicalize(self):
        """Reorders qubits to canonical ordering.

        This method ensures qubits are in the proper order by:
        1. Resharding final qubits that are sharded
        2. Computing the correct wire ordering
        3. Interchanging qubits until canonical order is achieved

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=4)
            >>> qdev.cnot(wires=[2, 3])
            >>> qdev.canonicalize()  # Ensure consistent ordering
        """
        # First maybe_reshard final self.log2_devices qubits
        sharded_wires = torch.nonzero(self._groupings[1] == -2).flatten()
        wires_to_unshard = sharded_wires[
            sharded_wires < (self.n_wires - self.log2_devices)
        ].tolist()
        do_not_shard = list(range(self.n_wires - self.log2_devices))
        for wire in wires_to_unshard:
            self.maybe_reshard(wires=[wire], ignore=do_not_shard)
        # Final qubits are all sharded (though possibly unordered); now obtain wire ordering
        wire_order = torch.nonzero(
            self._groupings[0] == 1
        ).flatten()  # First wire group is always ungrouped
        for i in range(2, self.num_dims - 1):
            group_mask = self._groupings[0] == i
            group_wires = torch.nonzero(group_mask).flatten()
            if len(group_wires) > 1:
                relative_order = self._groupings[1, group_mask].argsort()
                group_wires = group_wires[relative_order]
            wire_order = torch.cat((wire_order, group_wires))
        # Now interchange qubits until they are all in the proper order
        for i in range(self.n_wires):
            if wire_order[i] == i:
                continue
            else:
                if self._invertible_dummy is not None:
                    self._invertible_dummy, _ = self.interchange_qubits(
                        self._invertible_dummy, self._groupings, i, wire_order[i]
                    )
                self._states, self._groupings = self.interchange_qubits(
                    self._states, self._groupings, i, wire_order[i]
                )
                wire_order[wire_order == i], wire_order[i] = wire_order[i], i

    @property
    def states(self):
        """Returns the canonicalized quantum states.

        This property automatically canonicalizes qubit ordering before returning
        the state tensor.

        Returns:
            The quantum state tensor in canonical order.

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=2)
            >>> qdev.h(wires=[0])
            >>> print(qdev.states.shape)  # (1, 2, 2, 2)
        """
        self.canonicalize()
        return self._states

    @property
    def invertible_dummy(self):
        """Returns the invertible dummy tensor for backpropagation.

        Returns:
            The dummy tensor used for invertible operations, or None if not
            in invertible mode.

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=3, invertible=True)
            >>> dummy = qdev.invertible_dummy
            >>> # Use dummy for gradient computation
        """
        if self._invertible_dummy is not None:
            self.canonicalize()
            return self._invertible_dummy

    def maybe_reshard(
        self, wires: list[int], ignore: list[int] = [], inverse: bool = False
    ):
        """Reshards statevector if current sharding splits dimensions acted upon by gate.

        If the current sharding splits the statevector in dimensions that are acted upon
        by the gate, picks a new dimension to shard and redistributes accordingly.
        Incurs an all2all communication.

        Args:
            wires: Indices of qubits that will be acted upon by the gate. Typically
                contains at most two qubits.
            ignore: Indices of qubits that will not be sharded. Defaults to [].
            inverse: When going in reverse direction for invertible backpropagation,
                inverts the dimension picking logic. Defaults to False.

        Example:
            >>> qdev = DistributedQuantumDevice(n_wires=6, world_sz=2)
            >>> # Current sharding on wire 4
            >>> # Gate on wires (2, 4) may trigger resharding
            >>> qdev.maybe_reshard(wires=[2, 4])
            >>> # After resharding, wire 2 is sharded instead
        """
        if self.world_sz <= 1 or not wires:
            return
        cur_sharded_qubits = set(
            torch.nonzero(self._groupings[1] == -2).flatten().tolist()
        )
        overlap = set(wires) & cur_sharded_qubits
        if overlap:  # only if wires affect sharded dimensions
            new_qubit_sharding = cur_sharded_qubits - overlap
            usable_qubits = sorted(
                set(range(self.n_wires))
                - (set(wires) | cur_sharded_qubits | set(ignore))
            )
            # hardcode: 2qubit gates only
            min_wire = min(wires)
            max_wire = max(wires)
            # hardcode: n_wires > 2 * connectivity
            if max_wire - min_wire > min_wire + self.n_wires - max_wire:
                if inverse:
                    min_wire, max_wire = max_wire - self.n_wires, min_wire
                else:
                    min_wire, max_wire = max_wire, min_wire + self.n_wires
            # this happens to be the same for inverse and not!
            best_usable_qubits = [q_ for q_ in usable_qubits if q_ > max_wire] + [
                q_ for q_ in usable_qubits if q_ < min_wire
            ]
            uncoupled_qubits = (
                torch.nonzero(self._groupings[1] == -1).flatten().tolist()
            )
            # Select qubits to use in resharding
            if inverse:
                switch_qubits = best_usable_qubits[: len(overlap)]
            else:
                switch_qubits = best_usable_qubits[-len(overlap) :]
            already_uncoupled = list(set(uncoupled_qubits) & set(switch_qubits))
            needs_uncoupling = list(set(switch_qubits) - set(uncoupled_qubits))
            dummy_uncoupled = list(set(uncoupled_qubits) - set(switch_qubits))
            switch_qubits = already_uncoupled + needs_uncoupling
            # Move selected usable qubits into uncoupled dimensions (there are always at least 2 uncoupled qubits and at most 2 qubits selected)
            if needs_uncoupling:
                for i in range(len(needs_uncoupling)):
                    if self._invertible_dummy is not None:
                        self._invertible_dummy, _ = self.interchange_qubits(
                            self._invertible_dummy,
                            self._groupings,
                            needs_uncoupling[i],
                            dummy_uncoupled[i],
                        )
                    self._states, self._groupings = self.interchange_qubits(
                        self._states,
                        self._groupings,
                        needs_uncoupling[i],
                        dummy_uncoupled[i],
                    )
            # All switch qubits are now uncoupled
            for i in range(len(switch_qubits)):
                # Interchange uncoupled selected qubits and sharded qubits in anticipation of resharding
                if self._invertible_dummy is not None:
                    self._invertible_dummy, _ = self.interchange_qubits(
                        self._invertible_dummy,
                        self._groupings,
                        switch_qubits[i],
                        list(overlap)[i],
                    )
                self._states, self._groupings = self.interchange_qubits(
                    self._states, self._groupings, switch_qubits[i], list(overlap)[i]
                )
                (
                    self._groupings[1, switch_qubits[i]],
                    self._groupings[1, list(overlap)[i]],
                ) = (
                    -2,
                    -1,
                )
            new_qubit_sharding = new_qubit_sharding.union(switch_qubits)
            # all2all
            new_dim_sharding = [
                self._groupings[0, w].item() for w in new_qubit_sharding
            ]
            new_dim_sharding.sort()
            self._states = self._states.redistribute(
                self.device_mesh, placements=[Shard(d) for d in new_dim_sharding]
            )
            if self._invertible_dummy is not None:
                self._invertible_dummy = self._invertible_dummy.redistribute(
                    self.device_mesh, placements=[Shard(d) for d in new_dim_sharding]
                )


# Give DQD methods, so we can write e.g. `qdev.ry(wires=[0])`
for name_ in matrices.GATE_MAT_DICT.keys():
    func = partialmethod(getattr(functional, name_))
    setattr(DistributedQuantumDevice, name_, func)

__all__ = ["DistributedQuantumDevice"]
