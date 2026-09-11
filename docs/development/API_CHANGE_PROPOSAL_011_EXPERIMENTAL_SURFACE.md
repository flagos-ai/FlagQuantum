# API Change Proposal 011: Further Reduction of the Discoverable Experimental API

## Status

**Implemented, pending API-owner review — not frozen.**

- Target: first public alpha.
- Machine contract: `contracts/experimental-surface-v2-candidate.json`.
- Implementation authorization: the API owner explicitly requested further
  experimental API reduction on 2026-09-01.
- Proposal 008's frozen eight top-level domains and lifecycle rules remain unchanged.
- Stable APIs, Proposal 009's Interop protocol, and Proposal 010's DynamicCircuit
  contract remain unchanged.

## Problem

The first cleanup removed 99 flat `fq.experimental` names, but eight second-level
domains still exposed 114 feature symbols. Many were evidence records, rank/shard
state, low-level executors, kernel statistics, conformance reports, and P0-P5
research helpers. Autocompletion resembled an implementation index rather than
workflows users could try.

## Decision

Retain only task-level workflows and third-party adapter namespaces in the
discoverable experimental API:

| Domain | Previous count | Current count | Retained boundary |
| --- | ---: | ---: | --- |
| distributed | 25 | 2 | Distributed training workflows. |
| dynamic | 22 | 2 | Dynamic circuit execution and backend evaluation. |
| execution | 1 | 0 | Domain retained; use `fq.run`. |
| interop | 2 | 2 | Qiskit/PennyLane adapter namespaces. |
| mps | 9 | 0 | Domain retained; use the standard MPS backend. |
| numerics | 50 | 3 | Basic split real/imag execution and gradients. |
| planning | 3 | 0 | Domain retained; use `fq.plan`. |
| simulation | 2 | 1 | TEBD workflow. |

The discoverable feature surface falls from 114 to 10, removing 104 names (91.2%).
Workflows still return their records normally; users need not separately import
implementation record types from experimental namespaces.

## Migration Principles

Names removed from `__all__` and `dir()` are not stable APIs. Internal tests,
benchmarks, evidence tools, and implementation code now import their owning
modules directly rather than depending on experimental facades. Old internal
paths are removed; accessing names outside the v2 contract raises `AttributeError`.

The capability matrix now distinguishes public experimental entries from internal
development evidence. P2-P5 numerical research and conformance runners retain
code, tests, and evidence but no longer appear as public APIs in generated docs.

## Acceptance Criteria

- [x] Eight top-level domains remain unchanged.
- [x] Second-level discoverable exports exactly match the v2 machine contract.
- [x] Evidence, low-level state, result records, and conformance types leave autocompletion.
- [x] Stable root and approved stable extensions remain unchanged.
- [x] Internal callers leave nonpublic compatibility paths.
- [x] Temporary internal resolution paths are removed.
- [ ] API owner reviews the final first public alpha discoverable experimental surface.
