# Compiler

Transform quantum programs while preserving their numerical meaning,
measurement behavior, and required gradients. The design direction is one
verified path from quantum-classical source to target-legal output.

Compiler owns program transformations over Core semantics. Runtime chooses
execution; Simulation supplies numerical kernels; Compute and Remote adapt
resources. Compiler does not run the program.

## Start here

From the repository root:

```bash
python -m examples.compiler_optimize
python -m examples.target_aware_compilation
```

The first example checks optimization against the original circuit. The second
checks routing legality and numerical equivalence on a concrete topology.
Use `optimize(program)` for target-independent optimization and
`compile(program, coupling_map=...)` for target-aware compilation.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Canonical optimization | [pipeline.py](pipeline.py) |
| Connectivity and routing | [routing.py](routing.py), [sabre.py](sabre.py), [topology_legalization.py](topology_legalization.py) |
| Wire layouts and the layout restore | [layout.py](layout.py) |
| Native-gate and target requirements | [native_gate_legalization.py](native_gate_legalization.py), [target_legalization.py](target_legalization.py) |
| Dependency scheduling | [schedule_legalization.py](schedule_legalization.py) |
| Emission and round-trip checks | [target_emission.py](target_emission.py), [target_conformance.py](target_conformance.py) |
| Structured hybrid programs | [_hybrid/](_hybrid/README.md) |

A transformation is acceptable when it preserves the relevant state,
measurement, and gradient references, produces legal output, and has bounded
code growth. An optimization must not remove a trainable gate solely because
its present angle is zero.

Routing strategies are one boundary with several implementations, so a new or
replaced strategy is accepted only when the shared conformance suite in
[test_routing_conformance.py](../../tests/team/compiler/test_routing_conformance.py)
passes over it. That suite also runs an independent router and a replacement
router through the compiler, the topology legalizer, and the dynamic-circuit
runtime, which is how the boundary is shown to be replaceable rather than
merely defined.

[Implementation details](IMPLEMENTATION.md) document stage ownership and
migration constraints. [Testing policy](../../docs/development/TESTING.md)
defines the additional checks for each change.
