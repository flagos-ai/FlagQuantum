# Ecosystem Team Boundary

This package owns the extension manifest, capability negotiation, task-local
registry, lifecycle isolation, and conformance entry points. Preserve this SDK
as the only extension registry. Domain implementations remain in Compiler,
Runtime, Simulation, Compute, or Remote and must not move into this package.
