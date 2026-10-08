# IR

`flagquantum/core/ir/` owns FlagQuantum IR: the canonical, versioned, serializable
description of a quantum program, plus -- behind that public surface -- the
internal level boundary the compiler crosses when it needs a value flow and a
control-flow graph.

## What this package owns

- `__init__.py`: the public IR. `CircuitIR` (`kind`
  `flagquantum.circuit_ir`, schema `1.0`), its validation, and its
  `to_dict`/`from_dict`/`to_json`/`from_json` serialization. `IR_VERSION` is
  `"1.0"`.
- `levels.py`: the internal boundary. It states the vocabulary the internal
  level contract is written in and delegates the two conversions. It is not
  exported from `flagquantum` and does not appear in `docs/public_api_v1.json`.
- `diagnostics.py`: the refusal vocabulary -- the 28 diagnostic codes, the four
  level literals, `LevelContractError`, and `LevelDiagnostic`. One code is
  declared in exactly one place.
- `program/`: the program level, one file per responsibility. `model.py` is the
  datatype set (blocks, terminators, edges, value references, operations,
  records, conversions, opcode signatures). `verifier.py` is the type and
  control-flow rules. `lowering.py` is the two conversions. `__init__.py`
  re-exports the four.

## What this package must not own

- No second IR. `CircuitIR` schema `1.0` is the only serialized IR this product
  publishes, and `flagquantum/core/ir/__init__.py` is the only module under this
  package that is public.
- No second operator vocabulary. Opcode names, arities, and parameter names come
  from `flagquantum/core/operator_schema.py`; the program level derives its
  signatures from that schema rather than restating it.
- No pass infrastructure. There is one pass manager and it lives in
  `flagquantum/compiler/pass_manager.py`; nothing here defines a pass, a pass
  registry, an analysis manager, or a second artifact boundary.
- No execution. This package does not plan, schedule, simulate, or touch a
  target. It translates between two static descriptions of a program.

## Allowed dependencies

Only `flagquantum/core/` and the standard library. The package must not import
Compiler, Runtime, Simulation, Compute, Remote, ecosystem adapters, or any vendor
SDK. `tools/check_architecture.py` enforces this.

## Public entry points

`flagquantum.core.ir.CircuitIR` and `flagquantum.core.ir.IR_VERSION`, re-exported
as **`fq.Circuit`'s** `.to_ir()` / `fq.CircuitIR.from_dict()` round trip. Nothing
under `levels.py`, `diagnostics.py`, or `program/` is public, and none of those
names appear in the Stable Core snapshot.

## The internal contract

The seven rules this boundary must keep -- level entry and exit, value identity,
linearity, joins, failure categories, round trip, and rejection -- are declared in
`contracts/multi-level-ir-internal-v1-candidate.json`, read back by
`tools/check_multi_level_ir_contract.py`, and exercised by
`tests/unit/test_multi_level_ir_contract.py`. The gate reads every `.py` under
this package except the public `__init__.py`, so a refusal that moves into a
level package is still checked against the declared vocabulary.

Read `docs/architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md` and
`docs/development/API_CHANGE_PROPOSAL_062_MULTI_LEVEL_IR_PHASE_4_ACTIVATION.md`
before widening this boundary.

## Shortest path for a typical change

A representative small change is *rejecting one more malformed program*. Add the
code, raise it, declare it, and let the gate confirm both directions.

1. Declare the code in `diagnostics.py` and add it to the contract's
   `diagnostic_codes` list.
2. Raise it from the rule that owns the condition in `program/verifier.py` (or
   from `program/lowering.py` if the condition is only visible during
   conversion).
3. Add one case to `tests/unit/test_multi_level_ir_contract.py`.
4. Run:

```bash
python tools/check_multi_level_ir_contract.py
python -m pytest tests/unit/test_multi_level_ir_contract.py -q
python tools/ci_tier.py pr-runtime
```

`tools/check_multi_level_ir_contract.py` fails if a declared code is never
raised somewhere in the boundary, if a raised code is not declared, if a code's
declared level disagrees with the level the boundary raises it at, or if a
recorded `flagquantum.circuit_ir` payload no longer survives a round trip. The
contract's rules and this package's refusal sites are therefore kept in step by
the gate rather than by review.

Changing `CircuitIR` itself -- its schema, its fields, or `IR_VERSION` -- is not
this path. That is a protected public change: read
`docs/development/PUBLIC_API_PROTECTION.md`, get an approved proposal, and treat
the snapshot in `docs/public_api_v1.json` as an output of the change rather than
something to edit.
