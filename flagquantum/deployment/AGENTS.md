# Deployment Contract Boundary

This directory owns target-neutral packaging and task contracts. External target
submission, status, result decoding, and provider evidence stay behind Core-owned
execution contracts. Do not redefine public semantics or add general Compiler
passes and simulation algorithms.
