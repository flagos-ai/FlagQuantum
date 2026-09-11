# Python engineering standard

This standard applies to production code, tooling, tests, examples, and notebooks.
The goal is code that another engineer can understand, verify, and change safely.
Existing violations are migration work, not approved exceptions.

## Language

Use English throughout the repository, including comments, docstrings, errors,
logs, documentation, fixtures, notebook output, diagram labels, and filenames.
Translate technical meaning, decisions, qualifications, and historical status
faithfully. Preserve identifiers, equations, links, numerical results, and code
behavior. Do not delete documentation or conceal text using Unicode escapes.
Changes to hash-bound evidence require the existing evidence review process;
translation does not authorize presenting a changed artifact as the original.

## 1. Readable Python

Prefer direct iteration, comprehensions, unpacking, context managers, and suitable
standard-library types. Use `defaultdict`, `itertools`, and assignment expressions
when they make the intent clearer. Avoid clever expressions, unnecessary indexing,
star imports, redundant wrappers, and comments that restate the code.

## 2. Precise types

Annotate every function's parameters and return value, except conventional
`self` and `cls`. Use unions, `Literal`, protocols, `TypedDict`, and generics to
describe actual constraints. Use `Any` only at an unavoidable external boundary
and validate values before they enter internal code. Do not use casts or ignores
to hide a missing contract. Strict mypy or pyright checks are the acceptance
target for the complete Python surface; passing a subset is not whole-repository
certification.

## 3. Focused functions

Keep functions focused on one responsibility and one level of abstraction.
Prefer functions that fit on one screen. Review longer functions for separable
responsibilities; do not split numerical algorithms into meaningless forwarding
helpers merely to meet a line count. Prefer explicit inputs and return values
over mutation of caller-owned or global state.

## 4. Errors and resource ownership

Validate original values before coercion. Catch expected exceptions precisely,
retain causes, and provide actionable domain context. Never silently swallow
failures. A broad catch is justified only at a documented plugin, process, or
cleanup boundary that preserves or reports the failure. Use context managers
and `finally` to release files, sessions, locks, devices, and worker processes.
Keep the protected public exception contract intact.

## 5. Bounded memory

Stream large inputs and outputs. Prefer iterators and generators when consumers
do not need materialization; use `yield from` for natural delegation. Use lists
when repeated access, vectorization, or an API requires them. Document the memory
cost of dense conversions and avoid hidden full-state materialization.

## 6. Explicit dependencies

Pass clients, configuration, and services through existing constructors or
arguments. Avoid mutable global application state. Reuse existing typed settings
and lifecycle boundaries. Introduce Pydantic or another dependency only when a
verified requirement warrants it; validation must not create a parallel
configuration system. Tests should supply small fakes through those boundaries.

## 7. Measured performance

Choose appropriate algorithms and data structures before micro-optimizing.
Avoid repeated immutable-string concatenation and unnecessary work in hot loops.
Cache only when identity, invalidation, memory growth, and concurrency are clear;
do not transparently cache changing I/O results. Local binding and other small
optimizations require measurements when they reduce readability. Keep blocking
I/O off event-loop threads and introduce concurrency only for a concrete need.

## 8. Tests that protect behavior

Use pytest fixtures, independent references, boundary cases, failure scenarios,
and meaningful end-to-end examples. Do not substitute tests of wording, function
existence, or copied implementation logic for behavior. Use snapshots for stable
serialization contracts and controlled benchmarks for important performance
paths. Aim for greater than 95% meaningful coverage, including branches, while
reporting the exact measured scope. Coverage alone does not prove correctness.

## 9. Data with explicit structure

Prefer immutable records for configuration, requests, and results. Remember that
`frozen=True` does not freeze nested lists or dictionaries. A fixed-schema mapping
with more than three keys should have a `TypedDict`, dataclass, or suitable model;
arbitrary-key collections remain mappings. Reuse authoritative records rather
than creating duplicate envelopes. Name constants when a number has domain
meaning; literal arithmetic identities do not need invented configuration names.

## 10. Automated engineering and useful documentation

Document public modules, classes, and functions in English using Google-style
docstrings consistent with the repository. Explain arguments, results, and
relevant exceptions without mechanically repeating annotations. Provide a small
executable example for public workflows.

Use Black for formatting, Ruff for linting and import order, and mypy for static
checking through the existing pre-commit and CI workflows. Ruff already covers
the relevant isort/flake8 responsibilities; do not add duplicate tools solely
to increase the tool count. Use structured logging for library/runtime operational
events, with a configured JSON formatter where machine consumption is required.
Human-facing CLI output and teaching examples may print their intended results.
Do not install global logging handlers or a logging dependency at import time.

## Review and acceptance

For each change, review correctness, readability, ownership, type coverage,
complexity, memory, errors, and test relevance. Run the smallest meaningful tests,
then the tiers required by the change. Record unresolved checks accurately.
Neither an English-only policy nor a strict-check command establishes compliance
until the complete applicable scan and tests pass.
