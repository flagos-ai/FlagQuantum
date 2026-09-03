# Compiler Team Boundary

Compiler transforms Core-owned program artifacts through validation, analysis,
passes, lowering, and code generation. It must not submit jobs, manage devices,
implement simulation kernels, or expose provider SDK objects. Shared types must
move to Core through an integration contract change.
