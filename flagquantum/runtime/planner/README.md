# Runtime planner

This directory is the authoritative home for execution-mode and backend
selection, resource estimates, topology-aware candidate construction, noisy
backend selection, calibration inputs, and the final Runtime selection result.

It does not transform programs, execute numerical kernels, manage provider
lifecycle, or own the stable serialized `ExecutionPlan`. The package entry
compiles through the existing Compiler boundary and assembles the existing plan
product from resolved Runtime policy. Plan types and serialization live in
`runtime/execution_plan.py` and `runtime/execution_plan_contract.py`.

## Ten-minute change path

- Change stable `fq.plan` orchestration in `__init__.py`.
- Change backend cost decisions in `backend_selection.py`.
- Change memory estimates and execution policy in `estimates.py` and
  `execution_policy.py`.
- Change candidate evidence in `candidate_plans.py`, `candidates.py`, and
  `providers.py`.
- Change selection ranking in `selection_result.py`.
- Change noisy-backend decisions in `noise_selection.py` and their calibration
  schema in `noise_calibration.py`.
- Change final stable/noisy plan assembly in `__init__.py`.

Run the backend-selection, selection-context/result, execution-policy,
execution-plan, and CPU vertical-slice tests. An ordinary policy change should
remain in this directory.
