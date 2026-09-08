"""High-throughput Z/ZZ objective scans for distributed MPS reverse mode."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....simulation.mps.observables import mps_z_zz_local_scan
from ....simulation.mps.reverse import mps_vjp
from .reverse_observables import (
    SiteShardedZZScanResult,
    mps_expectation_and_adjoints,
    parse_mps_z_zz_terms,
)
from .reverse_transport import receive_reverse_tensor, send_reverse_tensor
from .state import RankOwnedMPSState

_recv = receive_reverse_tensor
_send = send_reverse_tensor


def mps_multi_observable_mse_and_adjoints(
    state: RankOwnedMPSState,
    terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]],
    *,
    compiled_observables: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    if not terms:
        raise ValueError("observable_terms must contain at least one MSE term")
    parsed = parse_mps_z_zz_terms(state, terms)
    if parsed is not None:
        return mps_fused_z_zz_mse_and_adjoints(
            state, parsed, compiled_observables=compiled_observables
        )
    total = torch.zeros(
        (),
        dtype=next(iter(state.local_tensors.values())).real.dtype,
        device=next(iter(state.local_tensors.values())).device,
    )
    adjoints = {
        wire: torch.zeros_like(tensor) for wire, tensor in state.local_tensors.items()
    }
    scale = 1.0 / len(terms)
    for observable, target in terms:
        value, term_adjoints = mps_expectation_and_adjoints(state, observable, target)
        total = total + scale * value
        for wire, derivative in term_adjoints.items():
            adjoints[wire] = adjoints[wire] + scale * derivative
    return total, adjoints


def mps_fused_z_zz_mse_and_adjoints(
    state: RankOwnedMPSState,
    parsed: Sequence[tuple[tuple[int, ...], torch.Tensor]],
    *,
    compiled_observables: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Evaluate all local Z/ZZ terms with one forward/reverse scan pair."""

    result = mps_site_sharded_z_zz_scan(
        state,
        parsed,
        compiled_observables=compiled_observables,
        compute_objective_adjoint=True,
    )
    assert result.loss is not None
    return result.loss, dict(result.adjoints)


def mps_site_sharded_z_zz_scan(
    state: RankOwnedMPSState,
    parsed: Sequence[tuple[tuple[int, ...], torch.Tensor]],
    *,
    compiled_observables: bool = False,
    compute_objective_adjoint: bool,
) -> SiteShardedZZScanResult:
    """Run the shared multi-channel scan used by training and observations."""

    if not parsed:
        raise ValueError("fused Z/ZZ objective requires at least one term")
    reference = next(iter(state.local_tensors.values()))
    targets = torch.stack([target for _, target in parsed], dim=-1)
    z_terms: dict[int, list[int]] = {}
    zz_terms: dict[int, list[int]] = {}
    for index, (wires, _) in enumerate(parsed):
        if len(wires) == 1:
            z_terms.setdefault(wires[0], []).append(index)
        else:
            zz_terms.setdefault(wires[1], []).append(index)

    local_inputs = local_outputs = local_variables = None
    carry = None
    with record_function("flagquantum::mps::left_environment_scan"):
        for owner in range(state.world_size):
            if state.rank == owner:
                if owner == 0:
                    base = torch.ones(
                        state.bsz, 1, 1, dtype=reference.dtype, device=reference.device
                    )
                    inputs = (base, torch.zeros_like(base)) + tuple(
                        torch.zeros_like(base) for _ in parsed
                    )
                else:
                    assert carry is not None
                    inputs = tuple(item.detach() for item in carry.unbind(0))
                inputs = tuple(
                    item.requires_grad_(compute_objective_adjoint) for item in inputs
                )
                variables = tuple(
                    state.local_tensors[wire]
                    .detach()
                    .requires_grad_(compute_objective_adjoint)
                    for wire in state.ownership[owner]
                )
                outputs = mps_z_zz_local_scan(
                    inputs,
                    variables,
                    state.ownership[owner],
                    z_terms,
                    zz_terms,
                    compiled=compiled_observables,
                )
                local_inputs, local_outputs, local_variables = (
                    inputs,
                    outputs,
                    variables,
                )
            if owner < state.world_size - 1:
                sequence = 2_500_000 + owner
                if state.rank == owner:
                    _send(
                        torch.stack([item.detach() for item in outputs]),
                        destination=owner + 1,
                        sequence=sequence,
                    )
                elif state.rank == owner + 1:
                    carry = _recv(reference, source=owner, sequence=sequence)

    assert (
        local_inputs is not None
        and local_outputs is not None
        and local_variables is not None
    )
    if state.rank == state.world_size - 1:
        differentiable_values = torch.stack(
            [torch.real(channel[:, 0, 0]) for channel in local_outputs[2:]], dim=-1
        )
        values = differentiable_values.detach().clone()
    else:
        differentiable_values = None
        values = torch.zeros(
            state.bsz, len(parsed), dtype=reference.real.dtype, device=reference.device
        )
    dist.broadcast(values, src=state.world_size - 1)

    if not compute_objective_adjoint:
        return SiteShardedZZScanResult(
            values=values,
            loss=None,
            adjoints={},
            forward_scans=1,
            reverse_scans=0,
            forward_messages=state.world_size - 1,
            reverse_messages=0,
        )

    output_grads = None
    with record_function("flagquantum::mps::objective_adjoint"):
        for owner in range(state.world_size - 1, -1, -1):
            if state.rank == owner:
                if owner == state.world_size - 1:
                    assert differentiable_values is not None
                    objective = (differentiable_values - targets).square().mean()
                    derivatives = mps_vjp(
                        (objective,),
                        local_inputs,
                        local_variables,
                        (torch.ones_like(objective),),
                    )
                    loss = objective.detach()
                else:
                    assert output_grads is not None
                    derivatives = mps_vjp(
                        local_outputs,
                        local_inputs,
                        local_variables,
                        output_grads,
                    )
                input_grads = tuple(
                    torch.zeros_like(value) if grad is None else grad
                    for value, grad in zip(
                        local_inputs, derivatives[: len(local_inputs)]
                    )
                )
                variable_grads = derivatives[len(local_inputs) :]
            if owner > 0:
                sequence = 2_600_000 + owner
                if state.rank == owner:
                    _send(
                        torch.stack(input_grads),
                        destination=owner - 1,
                        sequence=sequence,
                    )
                elif state.rank == owner - 1:
                    received = _recv(reference, source=owner, sequence=sequence)
                    output_grads = tuple(received.unbind(0))

    if state.rank != state.world_size - 1:
        loss = torch.zeros((), dtype=reference.real.dtype, device=reference.device)
    dist.broadcast(loss, src=state.world_size - 1)

    adjoints = {
        wire: torch.zeros_like(tensor) if grad is None else grad.detach()
        for wire, tensor, grad in zip(
            state.ownership[state.rank], state.local_tensors.values(), variable_grads
        )
    }
    return SiteShardedZZScanResult(
        values=values,
        loss=loss,
        adjoints=adjoints,
        forward_scans=1,
        reverse_scans=1,
        forward_messages=state.world_size - 1,
        reverse_messages=state.world_size - 1,
    )


def site_sharded_z_zz_objective_pipeline(
    states: Sequence[RankOwnedMPSState],
    terms: Sequence[Sequence[tuple[Mapping[int, str], torch.Tensor | float]]],
    *,
    max_pipeline_slots: int = 4,
    compiled_observables: bool = False,
    cancelled: Callable[[], bool] | None = None,
    on_window_drained: Callable[[int], None] | None = None,
) -> tuple[tuple[torch.Tensor, dict[int, torch.Tensor]], ...]:
    """Pipeline independent Z/adjacent-ZZ objectives across site shards.

    Slots retain their input order.  At most ``max_pipeline_slots`` objective
    graphs are live on a rank; a partial final window is drained normally.
    Cancellation is sampled collectively between windows, so no rank abandons
    an in-flight P2P operation. ``on_window_drained`` is a safe checkpoint or
    progress hook: it runs only after every rank has drained the window. A
    single slot deliberately takes the existing non-pipelined fast path.
    """

    if len(states) != len(terms):
        raise ValueError("states and terms must contain the same number of slots")
    if max_pipeline_slots < 1:
        raise ValueError("max_pipeline_slots must be positive")
    if not states:
        return ()
    first = states[0]
    if any(
        state.rank != first.rank
        or state.world_size != first.world_size
        or state.n_wires != first.n_wires
        for state in states
    ):
        raise ValueError("pipeline states must share rank ownership and wire count")
    parsed_slots = tuple(
        parse_mps_z_zz_terms(state, slot_terms)
        for state, slot_terms in zip(states, terms)
    )
    if any(parsed is None for parsed in parsed_slots):
        raise ValueError("pipeline supports only Z and adjacent-ZZ MSE terms")
    if len(states) == 1:
        return (
            mps_fused_z_zz_mse_and_adjoints(
                first, parsed_slots[0], compiled_observables=compiled_observables
            ),
        )

    results: list[tuple[torch.Tensor, dict[int, torch.Tensor]]] = []
    for window_start in range(0, len(states), max_pipeline_slots):
        window_states = states[window_start : window_start + max_pipeline_slots]
        window_parsed = parsed_slots[window_start : window_start + max_pipeline_slots]
        local_graphs = []
        for offset, (state, parsed) in enumerate(zip(window_states, window_parsed)):
            slot = window_start + offset
            reference = next(iter(state.local_tensors.values()))
            targets = torch.stack([target for _, target in parsed], dim=-1)
            z_terms: dict[int, list[int]] = {}
            zz_terms: dict[int, list[int]] = {}
            for index, (wires, _) in enumerate(parsed):
                (z_terms if len(wires) == 1 else zz_terms).setdefault(
                    wires[-1], []
                ).append(index)

            if state.rank == 0:
                base = torch.ones(
                    state.bsz, 1, 1, dtype=reference.dtype, device=reference.device
                )
                inputs = (base, torch.zeros_like(base)) + tuple(
                    torch.zeros_like(base) for _ in parsed
                )
            else:
                carry = _recv(
                    reference, source=state.rank - 1, sequence=4_000_000 + slot
                )
                inputs = tuple(item.detach() for item in carry.unbind(0))
            inputs = tuple(item.requires_grad_(True) for item in inputs)
            variables = tuple(
                state.local_tensors[wire].detach().requires_grad_(True)
                for wire in state.ownership[state.rank]
            )
            with record_function(f"flagquantum::mps::pipeline_forward_slot_{slot}"):
                outputs = mps_z_zz_local_scan(
                    inputs,
                    variables,
                    state.ownership[state.rank],
                    z_terms,
                    zz_terms,
                    compiled=compiled_observables,
                )
            local_graphs.append((state, targets, inputs, outputs, variables))
            if state.rank < state.world_size - 1:
                _send(
                    torch.stack([item.detach() for item in outputs]),
                    destination=state.rank + 1,
                    sequence=4_000_000 + slot,
                )

        window_losses: list[torch.Tensor] = []
        window_adjoints: list[dict[int, torch.Tensor]] = []
        for offset, (state, targets, inputs, outputs, variables) in enumerate(
            local_graphs
        ):
            slot = window_start + offset
            reference = next(iter(state.local_tensors.values()))
            if state.rank == state.world_size - 1:
                values = torch.stack(
                    [torch.real(channel[:, 0, 0]) for channel in outputs[2:]], dim=-1
                )
                objective = (values - targets).square().mean()
                derivatives = mps_vjp(
                    (objective,),
                    inputs,
                    variables,
                    (torch.ones_like(objective),),
                )
                loss = objective.detach()
            else:
                received = _recv(
                    reference, source=state.rank + 1, sequence=4_500_000 + slot
                )
                with record_function(f"flagquantum::mps::pipeline_reverse_slot_{slot}"):
                    derivatives = mps_vjp(
                        outputs,
                        inputs,
                        variables,
                        tuple(received.unbind(0)),
                    )
                loss = torch.zeros(
                    (), dtype=reference.real.dtype, device=reference.device
                )
            input_grads = tuple(
                torch.zeros_like(value) if grad is None else grad
                for value, grad in zip(inputs, derivatives[: len(inputs)])
            )
            if state.rank > 0:
                _send(
                    torch.stack(input_grads),
                    destination=state.rank - 1,
                    sequence=4_500_000 + slot,
                )
            variable_grads = derivatives[len(inputs) :]
            window_losses.append(loss)
            window_adjoints.append(
                {
                    wire: torch.zeros_like(tensor) if grad is None else grad.detach()
                    for wire, tensor, grad in zip(
                        state.ownership[state.rank],
                        state.local_tensors.values(),
                        variable_grads,
                    )
                }
            )

        losses = torch.stack(window_losses)
        dist.broadcast(losses, src=first.world_size - 1)
        results.extend(
            (losses[index], adjoints) for index, adjoints in enumerate(window_adjoints)
        )

        if on_window_drained is not None:
            on_window_drained(len(results))

        stop = torch.tensor(
            [1 if cancelled is not None and cancelled() else 0],
            dtype=torch.int32,
            device=next(iter(first.local_tensors.values())).device,
        )
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if int(stop.item()):
            break
    return tuple(results)


def site_sharded_z_zz_observations(
    state: RankOwnedMPSState,
    observation_sites: Sequence[int],
    *,
    compiled: bool = False,
) -> torch.Tensor:
    """Return every requested Z and adjacent ZZ value in two shard sweeps.

    The result is replicated as a small ``[batch, observable]`` tensor, while
    the MPS tensors remain site-owned.  Z columns precede adjacent-ZZ columns,
    matching ``MPSState.expectation_z_and_nearest_neighbor_zz``.
    """

    sites = tuple(dict.fromkeys(int(wire) for wire in observation_sites))
    if not sites or any(wire < 0 or wire >= state.n_wires for wire in sites):
        raise ValueError("observation_sites must contain valid MPS wires")
    parsed = tuple((wire,) for wire in sites) + tuple(
        (wire, wire + 1) for wire in sites if wire + 1 < state.n_wires
    )
    reference = next(iter(state.local_tensors.values()))
    zero = torch.zeros(state.bsz, dtype=reference.real.dtype, device=reference.device)
    scan = mps_site_sharded_z_zz_scan(
        state,
        tuple((wires, zero) for wires in parsed),
        compiled_observables=compiled,
        compute_objective_adjoint=False,
    )
    return scan.values


__all__ = (
    "mps_fused_z_zz_mse_and_adjoints",
    "mps_multi_observable_mse_and_adjoints",
    "mps_site_sharded_z_zz_scan",
    "site_sharded_z_zz_objective_pipeline",
    "site_sharded_z_zz_observations",
)
