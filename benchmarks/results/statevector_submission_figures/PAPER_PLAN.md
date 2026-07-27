# FlagQuantum MLSys Paper Plan

## Honest readiness assessment

FlagQuantum currently has a credible MLSys systems-paper core, but not yet a
main-track-ready evaluation. The distributed statevector backend is real:
sharded forward and backward, optimizer ownership, 1/2/4/8-GPU scaling,
two-node 16-GPU training, measured single-GPU OOM, multi-GPU capacity
completion, and raw-gradient correctness. The remaining gap is a unified final
evaluation and a sharper systems profile.

Current readiness estimate:

- system implementation: 80–85%;
- statevector evaluation: 65–75%;
- paper narrative and figures: 70%;
- reproducibility artifact: 50–60%;
- full FlagQuantum including optimized MPS/TN: not yet measurable as a paper
  claim.

## Recommended paper strategy

Use the title and architecture of the full FlagQuantum system, but make the
measured contribution explicitly “the first completed distributed backend:
differentiable statevectors.”

Do not wait for MPS/TN if the goal is a focused statevector systems paper.
Do wait if the intended claim is that FlagQuantum already provides three
production-grade distributed simulation modes.

## Planned final paper structure

1. Motivation: distributed forward alone is insufficient for quantum training.
2. Unified FlagQuantum IR and backend contracts.
3. Distributed differentiable statevector design.
4. Communication, topology, VJP, and optimizer synchronization optimizations.
5. Experimental methodology.
6. Same-host strong scaling.
7. Multi-node training and capacity expansion.
8. Statistical significance and external context.
9. MPS/TN architecture positions and, later, measured modules.
10. Limitations, related work, reproducibility, conclusion.

## Next figures and tables

### Required statevector additions

- Replace Figure 1 with one final-version 1/2/4/8/16 matrix, all with 30 samples.
- Add a system-profile figure:
  - kernel launches by phase;
  - GPU/SM activity;
  - compute and communication time;
  - achieved NVLink/RoCE bandwidth;
  - overlap and rank imbalance.
- Add a crossover heatmap over qubits, depth, and cross-shard-gate fraction.
- Add a compact hardware/software/reproducibility table.

### Reserved MPS figure

Four panels:

1. time and memory versus bond dimension;
2. accuracy versus exact statevector;
3. one-/multi-GPU crossover and capacity;
4. forward/backward/boundary-communication breakdown.

Minimum claim gate: sharded backward, boundary-adjoint exchange, optimizer
ownership, and one workload that cannot train on one GPU.

### Reserved TN figure

Four panels:

1. contraction/slicing plan and largest intermediate;
2. time and memory versus slice count;
3. reverse contraction and gradient accuracy;
4. single-/multi-node scaling with communication reductions.

Minimum claim gate: executed reverse contraction, explicit intermediate/slice
ownership, measured communication, and exact/approximate semantics.

## Submission decision gate

Submit to MLSys main track only when:

- the final-version 1/2/4/8/16 statevector matrix is complete;
- confidence intervals exclude no improvement for the headline workload;
- capacity OOM/completion remains reproducible;
- system-profile metrics explain the observed scaling;
- the artifact can regenerate primary tables and figures;
- related-work positioning identifies a clear novelty beyond “another GPU
  simulator.”

Until then, describe the work as a strong development system and paper draft,
not as an accepted-level result.
