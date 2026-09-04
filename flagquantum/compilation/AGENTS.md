# Transitional Planning Boundary

This directory temporarily contains execution-plan products, serialization,
assembly, and calibration. Stable program transformation, routing, and noise
lowering belong to `flagquantum/compiler`; execution selection and policy belong
to `flagquantum/runtime/planner`. Do not recreate them here. New cross-domain
types belong in Core only after an approved contract proposal.

The performance-calibration adapter may support the existing plan-product
method only. It must not select world size, backend, device, or execution mode.
