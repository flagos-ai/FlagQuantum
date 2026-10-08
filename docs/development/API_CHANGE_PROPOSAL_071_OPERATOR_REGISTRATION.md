# API Change Proposal 071: The custom-operation operand contract

## Status

**Proposed. No code is written by this proposal and no behaviour changes until the repository owner approves it and the acceptance items below are measured.**

Every measurement below was taken on commit `bc02d88b` on branch
`parity/w6-recovery`, in the authoritative worktree, with

```
$ cd fq-parity
$ export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD MPLCONFIGDIR=$PWD/.venv/mpl
$ .venv/bin/python -u /tmp/probe071.py
```

The probe's thirteen sections are cited by name. Four of them print more than
one measurement under one heading — section 5 (`5a`/`5b`/`5c`), section 8
(`8a`/`8a-prime`/`8b`/`8c`), section 12 (lowering, emitters, `fq.run`) and
section 13 (`13a`/`13b`/`13c`/`13d`) — and the subsection label is quoted
wherever that matters. The audit of what the shipped suites already exercise used a
second file, `/tmp/probe_any_plugin3.py`, loaded as a pytest plugin with

```
$ PYTHONPATH=$PWD:/tmp .venv/bin/python -m pytest -m "integration" -q \
    -p no:cacheprovider -p no:randomly -p probe_any_plugin3
```

and its output is in `/tmp/audit_integration3.txt`. The same plugin with no
marker filter and the paths `tests/unit` produced the unit-tier list in
`/tmp/probe_any_plugin.py`'s `/tmp/audit_run.txt`.

This proposal is a companion to
`docs/development/API_CHANGE_PROPOSAL_067_OPERATOR_MODIFIERS.md`, not a
replacement for it, and "Problem" section 14 measures exactly what 067 already
covers so that the two can be read as one design. The repository's
`docs/development/PUBLIC_API_PROTECTION.md` is the policy this document follows.

## Problem

### 1. A matrix is accepted on *any* opcode name, and afterwards the name alone decides the kernel

`Circuit.gate` accepts a `matrix=` keyword for every opcode, not only for the
opcode `any`:

```python
def gate(
    self,
    name: str,
    wires: Iterable[int] | int,
    *,
    params: Mapping[str, Any] | None = None,
    matrix: Any | None = None,
    **kwargs: Any,
) -> "Circuit":
```

There is exactly one refusal in that method, and it applies only to a declared
channel. Section 1 of the probe measures the acceptance:

```
  gate('h',[0], matrix=M)        opcode='h'       matrix stored: True
  gate('cx',[0,1], matrix=M4)    opcode='cx'      matrix stored: True
  gate('my_op',[0], matrix=M)    opcode='my_op'   matrix stored: True
  any(0, unitary=M)              opcode='any'     matrix stored: True
  gate('my_op',[0]) with no matrix  -> IRValidationError: unknown opcode 'my_op'; custom operations require an explicit matrix
  get_operator_schema('any')         -> None
  len(OPERATOR_SCHEMAS)              -> 35
  IR_VERSION                         -> '1.0'
  len(flagquantum.__all__)           -> 36
```

`flagquantum/core/ir/__init__.py:369` refuses an unknown name without a matrix and
accepts an unknown name *with* one, so the IR deliberately admits an arbitrary
named operation. The consequence, measured in "Problem" sections 2 to 5, is that the name
is what every executor dispatches on, and a supplied matrix is an input the
dispatch may simply not read.

This is not a documentation problem. The same call, `gate("z", [0], matrix=X)`,
produces `X|0>` through the default statevector executor, `Z|0>` through the
dynamic executor's diagonal kernel, and crashes through the dynamic entry point.
The three answers come from one program.

### 2. The default statevector executor never reads the operand for `x`, `y`, `cx` and `swap`

`flagquantum/simulation/statevector/local.py:838-845`:

```python
        name = canonical_opcode(instruction.name)
        if name in {"x", "cx", "swap"}:
            output = _apply_fixed_permutation(
                output, name, instruction.wires, circuit.n_wires
            )
        elif name == "y":
            output = _apply_single_qubit_fixed(
                output, name, instruction.wires[0], circuit.n_wires
            )
        else:
            matrix = _gate_matrix(...)
```

Neither branch tests `instruction.matrix`, and the `else` is the only path that
reads it. Section 3 of the probe, with `M` a seeded QR unitary whose image of
`|0>` is `[-0.1532, 0.9882]`:

```
  probe matrix M is not the named gate: M|0> = [-0.1532+0.0000j, 0.9882+0.0000j]
  gate('x',[0], matrix=M)        -> [0.0000+0.0000j, 1.0000+0.0000j]
  gate('y',[0], matrix=M)        -> [0.0000-0.0000j, 0.0000+1.0000j]
  gate('z',[0], matrix=M)        -> [-0.1532+0.0000j, 0.9882+0.0000j]
  gate('h',[0], matrix=M)        -> [-0.1532+0.0000j, 0.9882+0.0000j]
  any(0, unitary=M)              -> [-0.1532+0.0000j, 0.9882+0.0000j]
```

`z`, `h` and `any` return `M|0>`, so the matrix is honoured on the `else` path.
`x` returns `X|0>` and `y` returns `Y|0>`, so on those two names the operand is
discarded and the named gate is applied instead. The discard is silent: no
warning, no diagnostic, and the same `ExecutionResult` shape as a correct run.

**Negative control, patch and restore.** Adding the two-word guard
`and instruction.matrix is None` to both branches, running a *fresh* process,
and restoring the file, with the probe asserting `source.read_text() == backup`
afterwards. The `CONTROL` rows are the ones that matter for the guard's safety:
they prove that the same patch leaves every matrix-free path untouched, and that
it does not disturb the names and paths it must not touch.

```
  patching the two-word guard 'instruction.matrix is None' into the two branches:
    PATCHED x [-0.1532, 0.9882]
    PATCHED y [-0.1532, 0.9882]
    EXPECTED [-0.1532, 0.9882]
    CONTROL x  no matrix [0.0, 1.0]
    CONTROL y  no matrix [-0.0, 1.0]
    CONTROL cx no matrix [0.0, 0.0, 0.0, 1.0]
    CONTROL swap no matrix [0.0, 1.0, 0.0, 0.0]
    CONTROL any unitary [-0.1532, 0.9882]
    CONTROL z  matrix [-0.1532, 0.9882]
    CONTROL cx matrix [-0.1899, 0.1912, 0.4035, 0.8744]
    source restored: True
```

Read row by row: `x` with no matrix is still `|1>`, `y` is still `i|1>` (`y`'s
output is printed from the imaginary parts, because `Y|0> = i|1>`), `cx` on
`|10>` is still `|11>`, `swap` on `|10>` is still `|01>`, `any` with a unitary is
unchanged, `z` with a matrix was never on a guarded branch and is unchanged, and
`cx` **with** a matrix now takes the `else` path and returns `M4|10>`. So the
patch changes exactly the behaviour it names and nothing else.

The guard is sufficient and it is local: two branches, one two-word condition
each. The same program without the guard returns `[0.0, 1.0]` and `[0.0, 1.0]`
in the same position, which is the pre-patch value printed above.

`cx` and `swap` are covered by the same `if` and were not in section 3's fixture
list, so they are measured separately rather than asserted. The input is
prepared with `x` gates and not with `Circuit(inputs=...)`, so that both entry
points start from the same state and the two columns differ only in whether the
plan is executed. `cz` and `ccx` are the controls: neither is in
`{"x", "cx", "swap"}`, so both reach the `else` branch. Section 13a of the probe,
with the seeded 4x4 `M4` and 8x8 `M8`:

```
  DEFECT  prepared input = basis 2  want 4x4|input>=[-0.4514+0.0000j, 0.2476+0.0000j, -0.8261+0.0000j, 0.2290+0.0000j]
    cx    wires=(0, 1)    .state()=[0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j]   faithful=False
                          fq.run=[0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j]   agrees_with_state=True
    swap  wires=(0, 1)    .state()=[0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]   faithful=False
                          fq.run=[0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]   agrees_with_state=True
  CONTROL prepared input = basis 2  want 4x4|input>=[-0.4514+0.0000j, 0.2476+0.0000j, -0.8261+0.0000j, 0.2290+0.0000j]
    cz    wires=(0, 1)    .state()=[-0.4514+0.0000j, 0.2476+0.0000j, -0.8261+0.0000j, 0.2290+0.0000j]   faithful=True
                          fq.run=[-0.4514+0.0000j, 0.2476+0.0000j, -0.8261+0.0000j, 0.2290+0.0000j]   agrees_with_state=True
  CONTROL prepared input = basis 6  want 8x8|input>=[-0.2125+0.0000j, -0.1518+0.0000j, 0.3861+0.0000j, 0.0186+0.0000j, 0.1124+0.0000j, 0.6656+0.0000j, -0.5648+0.0000j, -0.0882+0.0000j]
    ccx   wires=(0, 1, 2) .state()=[-0.2125+0.0000j, -0.1518+0.0000j, 0.3861+0.0000j, 0.0186+0.0000j, 0.1124+0.0000j, 0.6656+0.0000j, -0.5648+0.0000j, -0.0882+0.0000j]   faithful=True
                          fq.run=[-0.2125+0.0000j, -0.1518+0.0000j, 0.3861+0.0000j, 0.0186+0.0000j, 0.1124+0.0000j, 0.6656+0.0000j, -0.5648+0.0000j, -0.0882+0.0000j]   agrees_with_state=True
```

`cx` on `|10>` returns `|11>` and `swap` on `|10>` returns `|01>` — the two fixed
permutations the branch is named for — while `want = M4|10>` has all four
amplitudes nonzero, so no permutation can be mistaken for it. `cz` and `ccx` in
the same position return the operand's image exactly. The two columns agree in
every row, which is the second point: `collapse_one_qubit_runs` folds one-qubit
runs only, so an arity-2 or arity-3 operand reaches the executor intact and the
`fq.run` divergence of "Problem" section 5 cannot appear here. This defect is
therefore purely in the executor's dispatch, and it is invisible to the planner.

**Re-measured.** An earlier census in this session concluded that the discard
was decided by the *preceding* instruction, and reported that `i` at arity 1 and
`cx` at arity 2 were the only affected names. That reading was an artefact of
the probe's own preparation, which happened to route around these fast paths.
The mechanism is the fast path, not the prefix, and the names are `x`, `y`, `cx`
and `swap`. The earlier premise is withdrawn.

### 3. The same shape of dispatch exists twice more, once guarded and twice not

`flagquantum/simulation/statevector/product_state.py:480-486` repeats the
pattern and adds a second name-keyed table:

```python
        if name in {"x", "cx", "swap"}:
            return _apply_fixed_permutation(
                component.state, name, local_wires, len(component.wires)
            )
        if enable_fixed_clifford and name in _FIXED_SINGLE_QUBIT_CLIFFORD_GATES:
            return _apply_fixed_clifford_gate(
                component.state, name, local_wires[0], len(component.wires)
            )
```

with `_FIXED_SINGLE_QUBIT_CLIFFORD_GATES = frozenset({"h", "s", "sdg", "y", "z"})`
at `:52`. Neither branch tests `instruction.matrix`.

The proposed guard is not new to this file. `product_state.py:116-122` is the
same kind of function written correctly:

```python
    if (
        instruction.matrix is not None
        or instruction.params
        or len(instruction.wires) != 1
        or name not in {"h", "s", "x"}
    ):
        return None
```

and `flagquantum/runtime/executors/statevector/reverse_adjoint_kernels.py:78`
is the same guard in the adjoint executor:

```python
    if instruction.name in {"cx", "swap", "x"} and instruction.matrix is None:
```

Section 11 prints both of those blocks from the source. There are therefore two
correct in-repo spellings and three defective call sites; the fix is to match
the existing spelling, not to invent a convention.

**Honest boundary on reachability.** The product-state executor is gated by
`product_state_execution_is_beneficial` and `_PRODUCT_STATE_MIN_WIRES = 16`,
and this session did not find an input that reaches it with a matrix-bearing
instruction: 16-, 20- and 24-wire circuits with a matrix-bearing `z`, `h`, `s`
or `y` all returned `product_state_calls=0`. The site is therefore reported as a
code-level defect of the same shape as "Problem" section 2, and *demonstrating that the
product-state executor is reachable with a matrix-bearing instruction* is an
acceptance item below, not a claim made here.

### 4. The identity-elimination pass deletes a matrix-bearing instruction before any executor sees it

`flagquantum/compiler/pipeline.py:84-88`, inside `remove_identity_gates`:

```python
        if instruction.name in {"i", "id"}:
            continue
        param_name = _ROTATION_PARAM.get(instruction.name)
        if param_name is not None and _is_zero(instruction.params.get(param_name)):
            continue
```

Neither test consults `instruction.matrix`. Section 4 of the probe:

```
  gate('i',[0], matrix=M)                        planned n=0 state=[1.0000+0.0000j, 0.0000+0.0000j]
  gate('id',[0], matrix=M)                       planned n=0 state=[1.0000+0.0000j, 0.0000+0.0000j]
  gate('rz',[0],params={'theta':0.0}, matrix=M)  planned n=0 state=[1.0000+0.0000j, 0.0000+0.0000j]
  gate('rx',[0],params={'theta':0.0}, matrix=M)  planned n=0 state=[1.0000+0.0000j, 0.0000+0.0000j]
```

`planned n=0` is the decisive number: the instruction is present in the IR and
absent from the execution plan, so this is a fourth *kind* of loss, upstream of
every executor. `gate("id", ...)` is additionally canonicalised to `i` before
the pass runs, so the alias and the canonical name are both affected.

The same loss reproduces through `fq.run` on the whole rotation family, which
matters because `_ROTATION_PARAM` at `flagquantum/compiler/pipeline.py:28-41`
maps twelve names to `"theta"` — `rx`, `ry`, `rz`, `phase`, `u1`, `rxx`, `ryy`,
`rzz`, `crx`, `cry`, `crz`, `cphase` — so the pass reaches arity-2 and
controlled names, not only arity-1 ones. Section 13b of the probe walks all
twelve at `theta = 0.0` with a matrix attached, gives each its own arity and wire
order, and reads the same circuit through both entry points. No input is seeded
on either side, so both columns start from `|0...0>` and differ only in whether
the plan is executed; the arity-1 rows carry the seeded 2x2 `M` of section 3 and
the arity-2 rows the 4x4 `M4` of section 13's header:

```
  rx      wires=(0,)      theta=0 planned n=0  .state()=[-0.1532+0.0000j, 0.9882+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
  ry      wires=(0,)      theta=0 planned n=0  .state()=[-0.1532+0.0000j, 0.9882+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
  rz      wires=(0,)      theta=0 planned n=0  .state()=[-0.1532+0.0000j, 0.9882+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
  phase   wires=(0,)      theta=0 planned n=0  .state()=[-0.1532+0.0000j, 0.9882+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
  u1      wires=(0,)      theta=0 planned n=0  .state()=[-0.1532+0.0000j, 0.9882+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
  rxx     wires=(0, 1)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.1912+0.0000j, 0.4035+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  ryy     wires=(0, 1)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.1912+0.0000j, 0.4035+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  rzz     wires=(0, 1)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.1912+0.0000j, 0.4035+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  cphase  wires=(0, 1)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.1912+0.0000j, 0.4035+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  crx     wires=(1, 0)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.4035+0.0000j, 0.1912+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  cry     wires=(1, 0)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.4035+0.0000j, 0.1912+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
  crz     wires=(1, 0)    theta=0 planned n=0  .state()=[-0.1899+0.0000j, 0.4035+0.0000j, 0.1912+0.0000j, 0.8744+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j]
```

Every `.state()` row is `M|0...0>`: the five arity-1 rows are `M|0>`, and
`M4|00> = [-0.1899, 0.1912, 0.4035, 0.8744]` is the four-amplitude row the four
two-wire families share, with the three controlled names swapping positions 1
and 2 because their wire order is `(1, 0)`. Every `fq.run` row is `|0...0>`
instead. So this defect is not arity-1-specific and not confined to `fq.run`'s
readout: the plan is empty in all twelve rows, and the correct amplitude is
reachable only through the entry point that does not execute the plan.

and the operand that was dropped is not small. With `M4` the `4 x 4` unitary
above and the circuit seeded to `|01>`, the two-wire rotation names plan to zero
instructions. In the block below the `planned n=` column is the plan's
instruction count and the `state=` column is `Circuit.state()`, which executes
the IR directly and so still applies the operand; the two columns are printed by
one probe line and are deliberately read together. `rxx` on wires `(0, 1)` gives
exactly `M4|01>`, and `crx`/`cry`/`crz` are built on wires `(1, 0)`, so their
three rows are the same image with the two middle amplitudes exchanged — the
endpoint-order convention, not a third behaviour:

```
  rxx     theta=0 matrix=M4  planned n=0  state=[-0.8556+0.0000j, -0.3466+0.0000j, 0.2953+0.0000j, -0.2463+0.0000j]
  ryy     theta=0 matrix=M4  planned n=0  state=[-0.8556+0.0000j, -0.3466+0.0000j, 0.2953+0.0000j, -0.2463+0.0000j]
  rzz     theta=0 matrix=M4  planned n=0  state=[-0.8556+0.0000j, -0.3466+0.0000j, 0.2953+0.0000j, -0.2463+0.0000j]
  cphase  theta=0 matrix=M4  planned n=0  state=[-0.8556+0.0000j, -0.3466+0.0000j, 0.2953+0.0000j, -0.2463+0.0000j]
  crx     theta=0 matrix=M4  planned n=0  state=[-0.4514+0.0000j, -0.8261+0.0000j, 0.2476+0.0000j, 0.2290+0.0000j]
  cry     theta=0 matrix=M4  planned n=0  state=[-0.4514+0.0000j, -0.8261+0.0000j, 0.2476+0.0000j, 0.2290+0.0000j]
  crz     theta=0 matrix=M4  planned n=0  state=[-0.4514+0.0000j, -0.8261+0.0000j, 0.2476+0.0000j, 0.2290+0.0000j]
```

The reference values are `M4|01> = [-0.8556, -0.3466, 0.2953, -0.2463]` and
`M4|10> = [-0.4514, 0.2476, -0.8261, 0.2290]`, so the `crx` row is `M4|10>` with
positions 1 and 2 transposed, which is what a `(1, 0)` wire list means. Every row
in the table is therefore the correct image of the operand. No row of that table
is what `fq.run` returns: `fq.run` honours the empty plan and returns
`|00>` in all seven rows, because the pass deleted the instruction before any
executor saw it. Two readings of one program, and the plan is the one that runs.

This is the one site where the correct behaviour is genuinely arguable in the
other direction: `i` with a matrix is unusual, and a reader could call it user
error rather than a defect. The measured fact is what decides it — the IR
accepts the program, the pass removes it, and the result is a silently different
program under the original name. Whether the fix is "keep it" or "refuse it" is
Decision 4 below; "delete it silently" is not a defensible third option.

**One observation recorded and not addressed.** A wrong-side operand is not
checked at construction ("Problem" section 7) but *is* caught eventually by the
executor's reshape, and the sizes involved are informative: a `(4, 4)` on one
wire fails as `RuntimeError: shape '[2, 2]' is invalid for input of size 16`, and
a `(2, 2)` on two wires as `RuntimeError: shape '[4, 4]' is invalid for input of
size 4`. So every wrong-side square is an error, only a late one — none is
silently accepted and silently wrong. That is why Decision 5 moves these earlier
rather than inventing a new rule, and why it leaves the three *flat* forms to
section 7: `(4,)`, `(1, 4)` and `(4, 1)` reshape successfully and do run, so they
are a different defect with a different remedy.

The asymmetry this proposal is actually about is between
`x`/`y`/`cx`/`swap` (operand discarded, silently, in the default executor) and
every other name (operand honoured, or an error). A proper "wrong side for this
arity" analysis would have to start from why
`flagquantum/simulation/gate_matrix.py:146` accepts a `(4,)` at all, and that is
"Problem" section 7's subject, not a fifth dispatch defect.

### 5. A run of two or more one-qubit gates rewrites a non-unitary operand into a normalised unitary

This is the finding that changes the shape of the whole contract, and it was not
in 067.

`collapse_one_qubit_runs` (`flagquantum/compiler/one_qubit_optimization.py:355`)
is the last pass of `OPTIMIZATION_PIPELINE`
(`flagquantum/compiler/pass_manager.py:45-55`, where it is the last of nine).
`_is_foldable` at `:294` returns
`True` for any single-wire instruction that carries a matrix, and `_emit` at
`:256` writes the product back as `u3` and `phase`/`rz`, whose angles come from
`_u3_angles` at `:234`:

```python
    theta = 2.0 * math.atan2(abs(bottom_left), abs(leading))
    phi = cmath.phase(bottom_left)
    lam = cmath.phase(-top_right) if abs(top_right) >= _PHASE_EPS else -phi
    return theta, phi, lam
```

and, under the docstring's own admission that a diagonal matrix is recognised
because "whatever survives there is float noise from composing the run", the
`_is_foldable` branch at `:302-303` that lets any matrix through:

```python
    if instruction.matrix is not None:
        return True
```

Every branch reads a *magnitude* into `atan2` or a *phase* into `cmath.phase`.
No magnitude survives into the emitted angles, so the emitted `u3` has unit
singular values whatever the operand had. Section 2 of the probe, operand
`diag(2, 1)` followed by `h`:

```
  program IR            ['h', 'any']
  planned IR            opcode='u3' params={'theta': 0.9272952180016121, 'phi': 0.0, 'lbd': 3.141592653589793} matrix=False
  plan.n_instructions   1
  diag(2,1) @ H         [1.4142+0.0000j, 1.4142+0.0000j, 0.7071+0.0000j, -0.7071+0.0000j]
  column-normalised     [0.8944+0.0000j, 0.8944+0.0000j, 0.4472+0.0000j, -0.4472+0.0000j]
  fq.run(...).state     [0.8944+0.0000j, 0.4472+0.0000j]
```

The emitted `u3` reproduces the column-normalised product and not the product.
Section 8 of the probe makes the boundary explicit, with the same six operands
on the same wire in three arrangements:

```
  8a. one matrix-bearing gate alone on the wire, input |+>: applied as written
    identity       state=[0.7071+0.0000j, 0.7071+0.0000j]  M|+>=[0.7071+0.0000j, 0.7071+0.0000j]  faithful=True
    2 * identity   state=[1.4142+0.0000j, 1.4142+0.0000j]  M|+>=[1.4142+0.0000j, 1.4142+0.0000j]  faithful=True
    diag(2,1)      state=[1.4142+0.0000j, 0.7071+0.0000j]  M|+>=[1.4142+0.0000j, 0.7071+0.0000j]  faithful=True
    diag(1,3)      state=[0.7071+0.0000j, 2.1213+0.0000j]  M|+>=[0.7071+0.0000j, 2.1213+0.0000j]  faithful=True
    shear          state=[1.4142+0.0000j, 0.7071+0.0000j]  M|+>=[1.4142+0.0000j, 0.7071+0.0000j]  faithful=True
    zeros(2,2)     state=[0.0000+0.0000j, 0.0000+0.0000j]  M|+>=[0.0000+0.0000j, 0.0000+0.0000j]  faithful=True
      a single instruction is never folded, so it is applied exactly
  8c. one gate versus two gates, |0> input
    identity       one gate -> [1.0000+0.0000j, 0.0000+0.0000j]   two gates -> [-0.0000+0.0000j, 1.0000+0.0000j]
    2 * identity   one gate -> [2.0000+0.0000j, 0.0000+0.0000j]   two gates -> [-0.0000+0.0000j, 1.0000+0.0000j]
    diag(2,1)      one gate -> [2.0000+0.0000j, 0.0000+0.0000j]   two gates -> [-0.0000+0.0000j, 1.0000+0.0000j]
    diag(1,3)      one gate -> [1.0000+0.0000j, 0.0000+0.0000j]   two gates -> [-0.0000+0.0000j, 1.0000+0.0000j]
    shear          one gate -> [1.0000+0.0000j, 0.0000+0.0000j]   two gates -> [0.7071+0.0000j, 0.7071+0.0000j]
    zeros(2,2)     one gate -> [0.0000+0.0000j, 0.0000+0.0000j]   two gates -> [1.0000+0.0000j, 0.0000+0.0000j]
```

One instruction is faithful for all six operands. Two instructions on the same
wire are faithful for none of `2*identity`, `diag(2,1)`, `diag(1,3)`, `shear`
or `zeros(2,2)`; `identity` is faithful only because the normalised product
happens to equal the true product when the operand is already unitary. So the
silent loss is **conditional on adjacency**, and a program that is correct in
isolation becomes wrong when an unrelated one-qubit gate is placed next to it.

Sections 8b and 8c read the state through `fq.run`, which is the planned path;
section 8a reads it through `Circuit.state()`, which is not. Section 13c of the
probe runs one circuit through both, with `h` then `any(diag(2, 1))` on `|0>`
and `h` then `h` as its own control, which isolates the divergence to the plan:

```
  h then any(diag(2,1))  .state()=[1.4142+0.0000j, 0.7071+0.0000j]   fq.run=[0.8944+0.0000j, 0.4472+0.0000j]
  h then h (control)     .state()=[1.0000+0.0000j, 0.0000+0.0000j]   fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
```

The control pair agrees to every printed digit, so the divergence is introduced
by the operand and not by the pipeline as such. Nothing in the executor
normalises and nothing in the readout normalises: the direct path is exact, and
the loss exists only on the path that runs the pass pipeline. That is why
Decision 2 repairs the pass rather than the executor, and why a test that
asserts on `Circuit.state()` would report this defect as absent.
That is the property that makes this a contract problem and not a rounding
problem.

**Re-measured.** This session first recorded the effect as "the runtime
renormalises a non-unitary matrix at readout". That was wrong, and "Problem" section 8 is
what corrects it:

```
  8a-prime. fq.run does not read Circuit(inputs=...); .state() does
    no instructions        .state()=[0.7071+0.0000j, 0.7071+0.0000j]  fq.run=[1.0000+0.0000j, 0.0000+0.0000j]
    one 2*identity gate    .state()=[1.4142+0.0000j, 1.4142+0.0000j]  fq.run=[2.0000+0.0000j, 0.0000+0.0000j]
```

`.state()` returns the unnormalised `2|0>`, so there is no renormalisation step
in the executor at all. The normalisation is inside the fold. The distinction
matters for the fix: a readout-side check would not address it, because the
information is already gone by the time anything is read out.

The 8a-prime asymmetry is not caused by anything in this proposal's subject — it
reproduces on a program with no custom matrix at all — and section 15 at the end
of "Problem" measures it on its own and records it as a separate, out-of-scope
gap. Two further things are worth stating about 8a while the fixture is in view.
First, `2 * identity` on `|+>` gives the unnormalised `[1.4142, 1.4142]`, which
is the control showing that no step in the direct path normalises anything.
Second, `zeros(2, 2)` gives the exact zero vector rather than an error, so
nothing in the direct path refuses a zero-norm program either; the only refusal
in the system for that operand is the fold's, cited in "Problem" section 9.

### 6. The dynamic executor dispatches on the name, and one name cannot complete a run

`flagquantum/runtime/dynamic/execution.py:99-116`:

```python
    if name in {"x", "cx", "swap"}:
        return _apply_fixed_permutation(state, name, instruction.wires, n_wires)
    if name == "y":
        return _apply_single_qubit_fixed(...)
    matrix = gate_matrix(...)
    apply_gate = (
        _apply_diagonal_matrix if name in _DIAGONAL_STATEVECTOR_GATES else _apply_matrix
    )
```

Section 5 of the probe measures both halves. At the kernel, with the operand as
the only input, input `|0>`:

```
  5a. one call to the dynamic kernel, operand as given, input |0>
    z   matrix=X      got=[0.0000+0.0000j, 0.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=False
    z   matrix=shear  got=[1.0000+0.0000j, 0.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=True
    s   matrix=X      got=[0.0000+0.0000j, 0.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=False
    s   matrix=shear  got=[1.0000+0.0000j, 0.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=True
    t   matrix=X      got=[0.0000+0.0000j, 0.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=False
    t   matrix=shear  got=[1.0000+0.0000j, 0.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=True
    rz  matrix=X      got=[0.0000+0.0000j, 0.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=False
    rz  matrix=shear  got=[1.0000+0.0000j, 0.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=True
    x   matrix=X      got=[0.0000+0.0000j, 1.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=True
    x   matrix=shear  got=[0.0000+0.0000j, 1.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=False
    h   matrix=X      got=[0.0000+0.0000j, 1.0000+0.0000j]  M|0>=[0.0000+0.0000j, 1.0000+0.0000j]  agrees=True
    h   matrix=shear  got=[1.0000+0.0000j, 0.0000+0.0000j]  M|0>=[1.0000+0.0000j, 0.0000+0.0000j]  agrees=True
```

For `z`, `s`, `t` and `rz` the kernel applies the diagonal of the operand and
drops the off-diagonal, so `X` supplied to `z` gives a **zero vector**. For `x`
it applies the fixed permutation. `h` is the only name whose kernel reads the
operand, and it does so only because `h` is not in either table.

Through the public entry point the same operand reaches an unhandled exception
rather than a wrong number:

```
  5b. the same operand through the public dynamic entry point
    matrix=X      opcode=z    RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
    matrix=X      opcode=s    RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
    matrix=X      opcode=x    ran
    matrix=X      opcode=h    ran
    matrix=X      opcode=any  ran
    matrix=shear  opcode=z    ran
    ...
  5c. the smallest reproduction of that crash
    DynamicCircuit(2).h(1).measure(1).gate('z',[0],matrix=X)
    run_dynamic(..., shots=4) -> RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
```

`run_dynamic` is reached from `flagquantum.experimental.dynamic`, and
`DynamicCircuit(2).h(1).measure(1)` is the smallest preamble that makes the
program genuinely dynamic. The `RuntimeError` is torch's, not FlagQuantum's: the
zero vector reaches `torch.multinomial` with no named error in between. This is
the only place in the probe where a matrix-bearing program produces an
exception, and it is a raw backend exception rather than a domain error.

`flagquantum/runtime/dynamic/execution.py` also holds the only `_apply_diagonal_matrix`
name-dispatch in the repository, so the same three-line guard has to be applied
in a third module, on its own terms: for the diagonal names the correct fix is
not "add a guard" but "stop choosing the kernel by name when a matrix is
present".

### 7. There is no construction-time check on the operand at all

`Instruction.__post_init__` (`flagquantum/core/ir/__init__.py:357`) checks the name, the
wires and the params, and never inspects `self.matrix` beyond testing it for
`None`. Section 6 of the probe:

```
  1 wire, (4,4)        CONSTRUCTED, stored shape (4, 4)
  2 wires, (2,2)       CONSTRUCTED, stored shape (2, 2)
  1 wire, (3,3)        CONSTRUCTED, stored shape (3, 3)
  1 wire, (2,3)        CONSTRUCTED, stored shape (2, 3)
  1 wire, (4,)         CONSTRUCTED, stored shape (4,)
  1 wire, (1,4)        CONSTRUCTED, stored shape (1, 4)
  1 wire, (4,1)        CONSTRUCTED, stored shape (4, 1)
```

Seven operands, including a matrix whose side is not a power of two and two that
are not square at all, all construct and store their shape verbatim.

The reason nothing downstream catches them is
`flagquantum/simulation/gate_matrix.py:144-147`, which reshapes rather than
validates:

```python
    if instruction.matrix is not None:
        matrix = getattr(instruction.matrix, "tensor", instruction.matrix)
        return torch.as_tensor(matrix, dtype=dtype, device=device).reshape(
            2 ** len(instruction.wires), 2 ** len(instruction.wires)
        )
```

`.reshape` is shape-agnostic about the source layout, so a `(4,)`, a `(1, 4)`
and a `(4, 1)` are all accepted as a 2x2, and no squareness is implied anywhere.
Section 7 of the probe measures both halves:

```
  1 wire, (4,4)        ExecutionError: planned execution failed
                         __cause__ RuntimeError: shape '[2, 2]' is invalid for input of size 16
  2 wires, (2,2)       ExecutionError: planned execution failed
                         __cause__ RuntimeError: shape '[4, 4]' is invalid for input of size 4
  1 wire, (3,3)        ExecutionError: planned execution failed
                         __cause__ RuntimeError: shape '[2, 2]' is invalid for input of size 9
  1 wire, (2,3)        ExecutionError: planned execution failed
                         __cause__ RuntimeError: shape '[2, 2]' is invalid for input of size 6
  1 wire, (4,)         RAN                     state=[1.0000+0.0000j, 0.0000+0.0000j]
  1 wire, (1,4)        RAN                     state=[1.0000+0.0000j, 0.0000+0.0000j]
  1 wire, (4,1)        RAN                     state=[1.0000+0.0000j, 0.0000+0.0000j]
```

Four of the seven fail at execution. The failure is
`ExecutionError("planned execution failed")`, raised at
`flagquantum/runtime/plan_execution.py:399` as
`raise ExecutionError(...) from error`, so the only way to learn what was wrong
is to read `__cause__`, which is a torch `RuntimeError` about a private reshape.
The other three **run**. They are not silently reinterpreted into a wrong
answer by accident; `.reshape` genuinely accepts them, and the resulting gate is
whatever the flat data spells in row-major order. A user who passes `(4,)`
because a previous call accepted `(2, 2)` gets a program that runs and a state
that has no declared relationship to the operand they passed.

### 8. A non-unitary operand is applied and then normalised, and a zero operand is input-dependent

Section 8 of the probe is quoted in "Problem" section 5 above for the adjacency effect.
The values are worth reading twice, because they show that the *only* check in
the system that would have refused `zeros(2, 2)` is a fold-specific one:

```
    2 * identity   one gate -> [2.0000+0.0000j, 0.0000+0.0000j]   two gates -> [-0.0000+0.0000j, 1.0000+0.0000j]
    zeros(2,2)     one gate -> [0.0000+0.0000j, 0.0000+0.0000j]   two gates -> [1.0000+0.0000j, 0.0000+0.0000j]
```

`zeros(2, 2)` on one gate yields the zero vector, and on two gates yields `|0>`,
because the fold drops a run whose product is not normalisable and the empty
replacement leaves the zero state. Two adjacent one-qubit instructions turn a
zero-norm program into a normalised basis state. Nothing reports it.

### 9. The one existing operand check is fold-specific, and the whole shipped suite exercises seven operands

`flagquantum/algorithms/folding.py:164-194` defines `_require_unitary`, called
at `:432-435`. Section 10 of the probe:

```
  fq.run(non-unitary 'shear')        -> [1.0000+0.0000j, 0.0000+0.0000j]
  fold_program(same circuit)         -> ValueError: the custom operation 'shear' on qubits (0,) is not unitary: the fold repeats it with its own conjugate transpose, which cancels only a unitary, and U^...
  fold_program((2,3) matrix)         -> ValueError: the custom operation 'wide' on qubits (0,) carries a matrix of shape (2, 3), which is not square and so has no conjugate transpose to fold with...
```

The first line is the problem stated as one fact: the same program is refused by
the fold and accepted by the runner.

The audit is what makes this proposal's blast radius knowable, and its result is
smaller than expected in one direction and larger in another. Two plugins were
used, because they answer different questions. `/tmp/probe_any_plugin.py` records
**every** non-channel custom-matrix instruction and classifies it as
`non-unitary`, `non-square`, `flat` or `wrong-dim`; run over `tests/unit` it
reported 48 distinct rows, of which **46 are `no-shape`** — a channel, whose
`matrix` is a Kraus tuple with no operand shape — and exactly **two are real
operands**:

- `tests/unit/test_folding.py::test_a_custom_operation_whose_matrix_is_not_unitary_is_refused`
  at `:692`, `opcode='shear'`, `shape=(2, 2)`, `wires=(0,)`, `non-unitary`, `x1`;
- the same test at `:716`, `opcode='wide'`, `shape=(2, 3)`, `wires=(0,)`,
  `non-square`, `x1`.

`/tmp/probe_any_plugin3.py` additionally captures the constructing call stack and
reports `gap = max|U^dagger U - I|`; run over the integration tier
(`-m "integration"`, `/tmp/audit_integration3.txt`) it reported **five**
operands, every one of them a `(2, 2)` `any` with `wires=` a single qubit:

- four parameterisations of `tests/test_native_circuit.py::test_dense_expectation_preserves_complex_precision_and_gradients`,
  `gap` `0.929736316204071` and `0.9297363256214735`, whose stack is
  `flagquantum/simulation/statevector/local.py:1203` →
  `flagquantum/circuit.py:427` → `flagquantum/circuit.py:415`;
- `tests/test_native_circuit.py::test_native_distributed_bridge_preserves_complex128_end_to_end`
  at `tests/test_native_circuit.py:1044`, `gap=1.0`.

That plugin deliberately prints only the suspicious classifications, so it does
not list a **unitary** operand. Reading `/tmp/audit_integration2.txt` (the same
plugin without the call-stack capture) shows the one it excludes: a `(4, 4)`
operand on two wires with `gap=1.0728836059570312e-06`, opcode
`pca_phase_power`, built at `flagquantum/algorithms/pca.py:381` by

```python
        circuit.any(control, *wires, unitary=controlled, name="pca_phase_power")
```

from `tests/test_algorithm_examples.py`'s guide-example test. So the whole
shipped corpus contains **seven** real custom-matrix operands — two in the unit
tier, five in the integration tier — plus this one genuinely two-wire unitary,
which is the only multi-wire custom operand in the repository and the only
existing coverage that an arity check must not break.

Consequently no case anywhere exercises `(4,)`, `(1, 4)`, `(4, 1)`, a
non-power-of-two side, or a square of the wrong arity. Any construction-time
check this proposal adds will be exercised for the first time by the tests this
proposal requires, and the disclosure in Compatibility below matters more than
usual. All eight operands are deliberate, and every one of them is a precision or
refusal probe rather than an accident — which is itself the finding: nothing in
9272 unit tests or the integration tier relies on the operand *contract*, so
nothing would have caught a change to it.

### 10. A declared channel shares `Instruction.matrix`, so an operand rule keyed on shape alone is wrong

Section 9 of the probe:

```
  opcode='depolarizing' channel=True
  instruction.matrix is a tuple of length 4
  element shapes [(2, 2), (2, 2), (2, 2), (2, 2)]
```

`instruction.matrix is not None` is true for a channel, and its value is a
4-tuple of Kraus operators, not a single operand. `gate_matrix`'s
`instruction.matrix is not None` branch is therefore unreachable for a channel,
and calling it directly raises
`ValueError: only one element tensors can be converted to Python scalars`.
The channel's Kraus tuple is derived from its declared parameters, and
`Circuit.gate` already refuses a channel that is given a matrix explicitly.

Any operand rule must therefore be keyed on "a matrix is present **and the
instruction is not a channel**". The classifier already exists:
`instruction.metadata["is_channel"]`, read at `flagquantum/core/ir/__init__.py:366`.

### 11. The dense-expectation helper deliberately routes a Hermitian, non-unitary operator through `Circuit.any`

This is the constraint that decides whether unitarity can be a construction-time
requirement, and the answer is that it cannot.

`flagquantum/circuit.py:884` defines a module-level helper

```python
def expectation(*ops: tuple[Any, Sequence[int]], ket: torch.Tensor) -> torch.Tensor:
    """Compute a small dense expectation value with native PyTorch tensors."""
```

and its implementation, `flagquantum/simulation/statevector/local.py:1203`,
builds a circuit whose every operand is the user's observable:

```python
    for matrix, wires in ops:
        circuit.any(*wires, unitary=matrix)
    return complex_mul(complex_conj(input_state), state(circuit)).sum(dim=-1)
```

The operator that
`tests/test_native_circuit.py::test_dense_expectation_preserves_complex_precision_and_gradients`
passes is Hermitian and **not** unitary:

```
op Hermitian: True
op unitary  : False
op norm^2   : 0.24477970600128174
```

so `U^dagger U` is `0.2448` away from the identity and the operator is carried
through `Circuit.any` on purpose. A construction-time rule of the form "a matrix
must be unitary" would refuse
`flagquantum.circuit.expectation`, a function that the integration tier
exercises in four parameterisations.

The two functions called `expectation` must not be confused. The stable root
export `fq.expectation` is the one in `flagquantum/observables/__init__.py` with the
signature `(observable: Observable, *, name: str | None = None) -> OutputRequest`
and is in the 36-name stable surface; `flagquantum.circuit.expectation` is the
dense helper above, is **not** a root export, and is nonetheless public enough
to be used by `tests/test_native_circuit.py`. `flagquantum.circuit`'s `__all__`
is `["Circuit", "expectation"]`, so it is not an accident.

The measured consequence is that the same `Circuit.any` carries two different
intents: "apply this matrix as a gate", and "carry this operator for a dense
expectation". A unitarity rule cannot be attached to the constructor without
choosing between them, and nothing in the current IR records which one the
caller meant.

### 12. Backend degradation is refused correctly but is not declared anywhere

Section 12 of the probe:

```
  pytorch          UnsupportedLoweringError: backend 'pytorch' cannot lower: any (not registered)
  jax              UnsupportedLoweringError: backend 'jax' cannot lower: any (not registered)
  mps              UnsupportedLoweringError: backend 'mps' cannot lower: any (not registered)
  tensor_network   UnsupportedLoweringError: backend 'tensor_network' cannot lower: any (not registered)
  qasm             UnsupportedLoweringError: backend 'qasm' cannot lower: any (not registered)
  qcis             UnsupportedLoweringError: backend 'qcis' cannot lower: any (not registered)
  qir              UnsupportedLoweringError: backend 'qir' cannot lower: any (not registered)
  provider         UnsupportedLoweringError: backend 'provider' cannot lower: any (not registered)
  named x + matrix, pytorch    ACCEPTED
  named x + matrix, qasm       ACCEPTED
```

All eight backends refuse the bare `any` opcode, which is the correct
fail-closed behaviour and is already tested. The last two lines are the gap: a
*matrix on a known name* passes `validate_lowering` for `pytorch` and for `qasm`,
because the registry is keyed on the opcode name and knows nothing about the
operand. The text exits then fail closed one layer later — `emit_openqasm`,
`emit_qcis` and `emit_qir` all raise
`OpenQASM 3.0 cannot represent the arbitrary matrix of instruction 0 ('x').`
through `require_static_gate_program` at
`flagquantum/compiler/operator_lowering.py:244` — and `fq.run` with a text
backend is refused earlier still, by
`CapabilityError: stable fq.run backend 'qasm' is not available`.

So the behaviour is correct at every exit and there is no single place that
states it. The parity-matrix row `custom_operation_registration` (in
`contracts/cudaq-parity-matrix.toml`) records this as its third absent half: a
declared backend-degradation contract for a user-defined operation. What is
missing is the declaration, not the refusal.

### 13. There are two constraints on *where* any of this can be added

**The install loop runs once, at import.** `flagquantum/circuit.py:972`:

```python
for _gate_name in sorted(set(OPERATOR_SCHEMAS) | set(OPERATOR_ALIASES)):
    _install_gate_method(_gate_name)
```

`_install_gate_method` at `:920` builds the per-opcode method from the schema's
arity and parameter names. A registration that adds a schema at runtime adds a
name to the table and **not** a bound method to `Circuit`, because this loop has
already run. A registration entry point therefore has to install the method
itself, or the registry and the `Circuit` surface will disagree for every name
added after import.

**`OPERATOR_SCHEMAS` is a public, versioned schema.** It is a
`MappingProxyType` of 35 entries and `flagquantum/core/AGENTS.md` states that
"Changes to protected artifacts, IR, public schemas, or Stable Core APIs require
an integration contract/ADR change before implementation." Any registration
design has to keep it immutable, which is what makes the registry pattern in 067
the right shape rather than an in-place mutation.

### 14. What proposal 067 already covers, and what this proposal is left holding

067 is titled "Control and adjoint modifiers, and an operator registration
table" and proposes `flagquantum.operators.{OperatorRegistry, register_operator}`,
`OperatorRegistry.schemas` / `.with_operator`, and `Circuit.control` /
`Circuit.adjoint`. Its sections 4 to 9 already establish the registration entry
point, the immutability of the schema table, the refusal of in-place
re-registration (`RE_REGISTER: operator schema 'h' is already registered`), the
namespace decision (`flagquantum.operators`, needing two coordinated edits to
`contracts/public-api-v1-candidate.json`), and the requirement that the
capability document be regenerated rather than hand-edited. Its measured probe
`/tmp/probe067c.py` already contains
`LOWERING_WITHOUT_CAPABILITY[qasm]: backend 'qasm' cannot lower: my_op (not registered)`
and `JSON_ROUND_TRIP_OPCODE: my_op`.

067 contains no shape, unitarity or construction-time operand premise at all. A
case-insensitive search of the document for `unitar`, `shape`, `reshape` and
`square` returns: the `_unitary` helper name (`:93`, `:95`, `:99`, `:103`,
`:696`), one rhetorical sentence on `:491` ("A unitary passed through
`Circuit.any` is not a modifier"), which is the closest 067 comes to this
subject; and three rhetorical uses of "shape" (`:187`, `:768`, `:968`). It
returns nothing for `reshape` and nothing for `square`.

This proposal is therefore scoped to the operand contract and the dispatch
defects, and it defers the registration table itself to 067 in full. Where the
two meet is "Problem" section 12 above: a registered operation needs a declared
degradation contract, which 067's registry can hold, but the operand it carries
is this proposal's subject.

### 15. A measured adjacent defect this proposal does **not** fix: `fq.run` ignores `Circuit(inputs=...)`

Section 8a of the probe reads a state through `Circuit.state()` rather than
through `fq.run`, and section 8a-prime states the reason as a caveat on the
fixture. That caveat is a general property, and it is recorded here because it
changes which symptom a user sees when they hit the operand defects above. It is
**out of scope for this proposal** and is not addressed by any Decision below.

`Circuit.__init__` accepts `inputs: torch.Tensor | None = None`
(`flagquantum/circuit.py:213`) and stores it in `self._inputs` and in
`circuit_param` (`:272`, `:278`), and `Circuit.adjoint` passes it through
(`:469`). The FlagQuantum IR has no initial-state field — a search of
`flagquantum/core/ir/__init__.py` for `inputs` returns nothing — and
`flagquantum/_api.py`'s `fq.run` accepts
`program_or_plan: Circuit | CircuitIR | ExecutionPlan`. Converting the `Circuit`
to IR therefore discards the declared initial state, and `fq.run` executes from
`|0...0>`. Section 13d of the probe measures it with **no custom matrix anywhere
in the program**, so that nothing in this proposal's subject can be blamed:

```
  |+> then h               .state()=[1.0000+0.0000j, 0.0000+0.0000j]   fq.run=[0.7071+0.0000j, 0.7071+0.0000j]
  |1> then h               .state()=[0.7071+0.0000j, -0.7071+0.0000j]   fq.run=[0.7071+0.0000j, 0.7071+0.0000j]
  |11> then h(0).cx(0,1)   .state()=[0.0000+0.0000j, 0.7071+0.0000j, -0.7071+0.0000j, 0.0000+0.0000j]   fq.run=[0.7071+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.7071+0.0000j]
```

`h` is self-inverse, so the first row is the whole result in one line: `|+>`
through `h` is `|0>`, and `fq.run` returns `h|0> = |+>` instead. The two-wire row
is the same loss on a Bell-shaped input: `.state()` gives the state `h(0).cx(0,1)`
produces from `|11>`, and `fq.run` gives the state it produces from `|00>`.

`fq.run`'s own docstring does not mention `inputs`, and no field of
`ExecutionOptions` (`mode, backend, device, target, batch_size, precision,
shots, seed, memory_limit_bytes, require_gradients, allow_approximate,
allow_backend_fallback`) carries an initial state, so there is no argument
through which a caller can restore it. `Circuit.state()` honours it, which is
why section 8a uses `.state()`.

Two consequences for this document, stated so that neither is discovered by a
reviewer as a surprise:

1. Every `fq.run(...)` figure in "Problem" sections 2 to 8 was taken on an input
   the program itself sets (a `|0>` register, or a gate sequence that produces
   the state being tested), never on `Circuit(inputs=...)`. Section 8a is the
   only place a declared input is used, and it reads `.state()`. Section 13a
   deliberately prepares its input with `x` gates for the same reason, so that
   its two columns are comparable.
2. This is a second, independent gap of the same class — a declared property of
   the program that a second entry point silently drops — and it needs its own
   proposal with its own fixtures. It is **not** covered by this proposal, is
   **not** covered by 067, and does not appear in the parity matrix's
   `custom_operation_registration` row. It is recorded here because it was found
   here, and it is a candidate for its own row.

## Decision

### 1. The rule is stated on the operand, not on the opcode name

A supplied matrix is honoured uniformly, or the program is refused. There is no
third answer, and today the system gives two different third answers depending
on which name and which executor the instruction reaches. The rule is:

> When an instruction carries a matrix and is not a declared channel, every
> executor applies that matrix and no name-keyed fast path may intercept it.

"Every executor" is the load-bearing phrase. The measured defects are in three
modules and five call sites, and fixing one of them leaves the others:
`flagquantum/simulation/statevector/local.py:838-845` and
`flagquantum/simulation/statevector/product_state.py:480-486` are corrected by
adding `instruction.matrix is None` to each name test, matching the two existing
correct spellings at `product_state.py:116-122` and
`reverse_adjoint_kernels.py:78`;
`flagquantum/compiler/pipeline.py:84-88` is corrected by skipping the
elimination when `instruction.matrix is not None`; and
`flagquantum/runtime/dynamic/execution.py:99-116` is corrected by removing the
name-keyed choice between `_apply_diagonal_matrix` and `_apply_matrix` when a
matrix is present, because a user-supplied operand is not diagonal merely because
the name is `z`.

The `local.py` fix has a measured negative control in section 2 of "Problem",
including a fresh-process rerun and a restored-file assertion, so it is the one
part of this decision that is already known to work.

### 2. `collapse_one_qubit_runs` refuses a non-unitary operand instead of normalising it

`_u3_angles` cannot represent a non-unitary matrix; the emitted angles are
magnitudes and phases only, so the fold is total only over unitaries. Section 4
of "Problem" measures the loss, and "Problem" section 5 measures that it is conditional on
adjacency, which means the same program is right or wrong depending on an
unrelated neighbouring gate.

Two repairs are possible: make `_is_foldable` return `False` for a
matrix-bearing instruction that is not unitary, so the run is left alone and
applied exactly; or make the fold refuse the program by name. The first is
chosen, because it preserves the single-instruction behaviour measured as
faithful in "Problem" section 8a and because the fold is an optimisation — an optimisation
that cannot preserve the program must decline to fire, not change the answer.
This is also the rule `one_qubit_optimization` already states for itself, in
`collapse_one_qubit_runs`'s own docstring: a run "is left untouched when it
carries a trainable angle, when a single z-rotation/pulse vocabulary already
spells it, or when folding it would not be strictly shorter". A non-unitary
operand is the same class of case.

The inconsistency is inside one package. `flagquantum/compiler/one_qubit_synthesis.py`
already answers exactly this question — `_UNITARY_ATOL = 1.0e-9` at `:51`,
`_is_unitary` at `:293`, and the refusal at `:462` —

```python
    if not _is_unitary(target):
        raise ValueError("matrix must be a 2x2 unitary")
```

while its sibling `one_qubit_optimization.py` folds the same shape of operand
with no such test. A reader who finds one of the two modules has no way to know
which answer the other gives.

The unitarity test must not be applied at construction, for the reason measured
in "Problem" section 11. It is a property the *fold* needs, and
`flagquantum/algorithms/folding.py:164-194` is the precedent for testing it
where it is needed.

### 3. `Circuit.any` keeps accepting a non-unitary operand; a new keyword records the intent

Because `flagquantum.circuit.expectation` routes a Hermitian, non-unitary
operator through `Circuit.any` on purpose, unitarity cannot become a
construction-time precondition. The intent is instead recorded explicitly, and
the operand contract depends on which intent was declared.

The shape of the decision, deliberately left as a decision rather than a
signature, is:

- a matrix carried as a *gate* must be an operand the executors can apply — a
  square matrix of side `2 ** len(wires)` — and is subject to the dispatch rule
  of Decision 1;
- a matrix carried as an *operator* for a dense expectation is exempt from
  squareness-of-side expectations only in the sense that it is still a
  `2 ** len(wires)` square, but is exempt from unitarity;
- the flat `(4,)`, `(1, 4)` and `(4, 1)` forms are **refused**, because "Problem" section 7
  shows they are accepted by `.reshape` and therefore run, and an accepted form
  with no declared meaning is the defect this proposal exists to remove. A
  caller who holds a flat buffer writes `.reshape(n, n)`.

The exact keyword name is left to the API owner. It is named here as an open
question rather than invented, because `PUBLIC_API_PROTECTION.md` bans
relative and transitional labels and because 067 owns the neighbouring
namespace.

### 4. The identity-elimination pass has to choose, and the choice is to refuse

`remove_identity_gates` eliminates `i`/`id` and zero-angle rotations because
they are semantic no-ops. With a matrix present, they are not no-ops: "Problem"
section 4 measures a matrix-bearing `i` producing `|0>` from a program that
should have produced `M|0>`, and the same for twelve rotation names at
`theta = 0.0`.

The pass has two defensible behaviours and the measured facts select one. It
cannot keep the instruction, because the pass exists to delete exactly that
shape and a matrix-bearing `i` is indistinguishable downstream from a genuine
one. It can refuse. Refusing is chosen, with a named error, so that the program
never reaches an executor under a name that misdescribes it. A caller who
genuinely wants `i` with a matrix has `Circuit.any`, which has no name-based
elimination.

### 5. A malformed operand becomes a named error at construction

Today a wrong-side operand fails at execution as
`ExecutionError: planned execution failed`, with the reason in `__cause__` as a
torch `RuntimeError` about a private reshape; a flat `(4,)`, `(1, 4)` or `(4, 1)`
does not fail at all and runs as whatever the flat data spells (Decision 3,
"Problem" section 7). `ExecutionError` is documented
elsewhere as the wrapper for a failed execution, and
`tests/team/runtime/test_execution_lifecycle_characterization.py:217` and
`tests/team/remote/test_workspace_executor.py:202,209` pin the string `planned
execution failed`, so the wrapper itself stays and is not renamed.

What changes is *when* the failure happens. A `(3, 3)` operand, or a `(4, 4)`
operand on one wire, is a malformed argument to `Circuit.gate` and belongs in
`IRValidationError`, raised from `Instruction.__post_init__` beside the existing
`unknown opcode {name!r}; custom operations require an explicit matrix`. The
error states the expected side and the received shape:

```
the custom operation 'wide' on qubits (0,) carries a matrix of shape (2, 3);
a single-wire operand must be a (2, 2) square
```

The message names the operand, the wires and both shapes, in the style of
`flagquantum/algorithms/folding.py`'s existing refusal.

### 6. Channels are explicitly out of scope, by classifier and not by shape

`instruction.metadata["is_channel"]` is the classifier
(`flagquantum/core/ir/__init__.py:366`), and section 10 of "Problem" measures why shape
cannot substitute for it: a channel's `matrix` is a Kraus 4-tuple of `(2, 2)`
tensors, so it is square, non-empty, and not an operand. Every rule in this
proposal reads the classifier first.

### 7. Backend degradation becomes a declared capability with a stated fallback

The refusals measured in "Problem" section 12 stay exactly as they are; no backend gains or
loses a refusal. What is added is the declaration: for a custom operation, each
backend states whether it can lower it, and a backend that cannot must say so
through the existing `LoweringCapability(backend, opcode, strategy, implementation, supported=True, reason="")`
record rather than only through the absence of a registry entry. The distinction
is observable: today the reason text for `any` is the generic `not registered`,
which is the same string a typo would produce.

067 owns the registry that holds these capabilities. This proposal requires only
that the operand decision be *visible* in it — a capability declared for an
opcode that receives a matrix must be able to say whether it accepts the operand
— and leaves the container to 067.

### 8. The registration entry point is 067's, and the late-install constraint is recorded

No second entry point is proposed. The constraint measured in "Problem" section 13 is
recorded here because it is a property of `Circuit` and of the install loop, not
of the registry: `flagquantum/circuit.py:972` runs once at import, so
`register_operator` has to install the bound method as part of registration or
the registry will hold an opcode that `Circuit` cannot spell. 067's acceptance
items do not currently cover that, and this proposal adds it as an item.

### 9. `OPERATOR_SCHEMAS` stays a `MappingProxyType` of 35 entries

No rule in this proposal adds, removes or mutates a schema entry. The operand
contract is a property of `Instruction`, the dispatch rule is a property of the
executors, and the declaration in Decision 7 lives in the lowering registry.
`len(OPERATOR_SCHEMAS) == 35` and `IR_VERSION == "1.0"` are unchanged, and both
are asserted in the acceptance list.

## Public API

This proposal adds **no root export**. `len(flagquantum.__all__)` is 36 today
and stays 36; `docs/public_api_v1.json`'s `stable_exports` is unchanged;
`experimental_stability` stays `"none"`; `IR_VERSION` stays `"1.0"`.

The only candidate-namespace change is the one 067 already needs. If the
intent-recording keyword of Decision 3 lands on `Circuit.gate` / `Circuit.any`,
that is a signature change on an existing stable surface and is governed by
`PUBLIC_API_PROTECTION.md`'s additive rules, not by this proposal's own scope.
The measured surface today is:

```python
Circuit.any(self, *wires: int, unitary: Any, name: str = "any") -> "Circuit"
Circuit.gate(self, name: str, wires: Iterable[int] | int, *,
             params: Mapping[str, Any] | None = None,
             matrix: Any | None = None, **kwargs: Any) -> "Circuit"
```

`Circuit.unitary = any`, so a keyword added to `any` is added to `unitary` too.

The behaviour this proposal changes is not reachable through a *new* name. It is
reachable through names that already exist, which is why the Compatibility
section below is longer than the API section.

## Compatibility

This is an implementation change to existing behaviour, not an addition of new
surface, so `PUBLIC_API_PROTECTION.md`'s first case applies: "No API proposal is
required when all protected surface, signature, semantic, and serialization
checks remain unchanged." They do **not** all remain unchanged, and this section
records each one that moves.

- **`IR_VERSION` is unchanged at `"1.0"`, and so is `len(OPERATOR_SCHEMAS)` at
  35.** Section 1 of the probe prints both. Nothing in this proposal adds an IR
  field or a schema entry.

- **`docs/public_api_v1.json` is unchanged, and its `stable_exports` list stays
  at 36 names.** `tools/public_api_snapshot.py` refuses automatic regeneration
  and reports `public API migration baseline passed`.

- **Three suites pin behaviour this proposal changes, and each is disclosed
  rather than quietly adjusted.**

  - `tests/unit/test_folding.py::test_a_custom_operation_whose_matrix_is_not_unitary_is_refused`
    **depends on the IR accepting** a non-unitary `(2, 2)` at `:692` and a
    non-square `(2, 3)` at `:716`, because it asserts the *fold* refuses them.
    Under Decision 5 the `wide` operand at `:716` becomes an
    `IRValidationError` at construction, so that half of the test changes from
    "the fold refuses it" to "the constructor refuses it". The `shear` half at
    `:692` is unaffected, because Decision 3 keeps a non-unitary operand legal
    at construction. This is the only place in 9272 unit tests where a
    non-square operand is built, so the change is one test, and it is a
    *strengthening*: today the check is unreachable from `fq.run`.
  - `tests/test_native_circuit.py::test_dense_expectation_preserves_complex_precision_and_gradients`
    and `test_native_distributed_bridge_preserves_complex128_end_to_end` build
    non-unitary `(2, 2)` operands through `Circuit.any`. Both keep working under
    Decision 3, which is the whole reason unitarity is not a construction-time
    precondition. The integration audit shows these five operands are the entire
    non-unitary corpus, so if Decision 3 is rejected in favour of a unitarity
    precondition, exactly these tests break and `flagquantum.circuit.expectation`
    becomes unusable. This is stated as a coupling, not a prediction.
  - `tests/test_algorithm_examples.py`'s guide-example test builds the only
    **two-wire** custom operand in the corpus — a unitary `(4, 4)` at
    `flagquantum/algorithms/pca.py:381`, `opcode='pca_phase_power'`, `gap=1.0728836059570312e-06`
    — from `tests/test_algorithm_examples.py`. It is the one existing case that
    proves a legitimately arity-2 operand still lowers, so the arity check of
    Decision 5 has to be measured against it and not only against the arity-1
    fixtures of "Problem" section 6.
  - `tests/team/runtime/test_execution_lifecycle_characterization.py:217`,
    `tests/team/remote/test_workspace_executor.py:202` and `:209` pin the string
    `planned execution failed`. Decision 5 moves a *class* of failures out of
    that wrapper and into `IRValidationError`; the wrapper and its string are
    not changed, so these three tests keep passing and are not edited.

- **The negative control for Decision 1 has been run.** Adding
  `and instruction.matrix is None` to `local.py:838` and `:842`, in a fresh
  process against a backed-up file, changes `gate('x',[0],matrix=M)` from
  `[0.0, 1.0]` to `M|0> = [-0.1532, 0.9882]`, and `gate('y',...)` likewise, while
  leaving `gate('x',[0])`, `gate('cx',[0,1])` and `any(0, unitary=M)` correct.
  The file is restored and the restoration is asserted. The guard is therefore
  known to fix the measured defect before the proposal is accepted.

- **A matrix-bearing `x` is currently accepted by `validate_lowering` for every
  backend** ("Problem" section 12). Decision 7 does not change that refusal set; it makes
  the acceptance visible as a capability record. No test currently asserts the
  acceptance, and this proposal adds one.

- **The parity row moves.** `contracts/cudaq-parity-matrix.toml`'s
  `custom_operation_registration` row is `status = "partial"`, `priority = "now"`,
  `dependency_class = "none"`, with an `evidence` list whose last entry is the
  `search:` token
  `no public callable that adds, replaces, or removes an operation and no declared backend degradation behaviour for a user-defined operation`.
  The second half of that token becomes false when Decision 7 lands, so the row's
  `reason` and that evidence entry are edited in the same change as the code, and
  `docs/generated/CAPABILITIES.md` is regenerated rather than hand-edited.
  `tools/parity_matrix.py --check` must still report
  `parity-matrix: contract valid and document current`.

- **`capability-maturity.toml` has no entry for this area, and one entry's
  `limitations` sentence has to be corrected.** Measured: the 87
  `[capabilities.*]` tables contain no `custom_operation_registration` block, so
  no maturity level moves and none is claimed. Two names that look like they
  belong to that file do not: `pass_manager_and_pass_plugins` and
  `compiler_plugin_ecosystem` are **row ids in `contracts/cudaq-parity-matrix.toml`**,
  and it is their `maturity_ref` fields that point into `capability-maturity.toml`
  — at `program_compilation` and `extension_sdk` respectively. The entry that
  actually owns the code Decision 2 and Decision 4 edit is
  **`program_compilation`** (`level = "production_supported"`, no `maturity_ref`
  of its own, `public_apis` including `flagquantum.compiler.optimize`), and its
  `limitations` asserts:

  > optimize() removes identity gates, merges self-inverse runs, and merges
  > adjacent rotations to a fixed point.

  The first clause is what "Problem" section 4 falsifies for a matrix-bearing
  instruction: the pass does remove `i`/`id` and zero-angle rotations, and it
  does so without reading `instruction.matrix`. Decision 4 makes that clause
  conditional, so the sentence is edited in the same change as the code rather
  than left as a totality claim the compiler does not honour.

  A separate entry, **`algorithms_gate_folding`** (`level = "development_evidence"`),
  owns the *probabilistic error cancellation* fold in
  `flagquantum/algorithms/folding.py` and is not the compiler fold; the two are
  unrelated code and must not be conflated when this proposal is reviewed. That
  entry's `limitations` already states the rule this proposal extends to the
  compiler — "a custom instruction whose matrix is not unitary or not square,
  because the fold repeats it with its own conjugate transpose and the identity
  would not hold" — and `flagquantum/algorithms/folding.py:164-194` implements
  exactly that refusal. So the repository documents and enforces the operand
  contract in one fold and neither documents nor enforces it in the other, which
  is the same asymmetry "Problem" section 10 measures, stated a second time in
  the maturity matrix.

- **A registered operation still cannot be spelled on `Circuit` after import**
  until the install-loop constraint of Decision 8 is addressed. That is not a
  regression introduced here; it is a measured property of `circuit.py:972` that
  this proposal records so that 067's acceptance list can carry it.

## Required evidence before this proposal can be accepted

Each item names the measurement, not the intention. A green suite does not show
that a refusal is exercised, so each refusal below is additionally proven by
deleting it and observing a test fail.

- [ ] The `local.py` guard is shown to fix `x`, `y`, `cx` and `swap`, not only
      `x` and `y`. The probe's negative control already covers the two
      single-wire branches ("Problem" section 2) and section 13a already measures
      `cx` and `swap` at arity 2 against a `(4, 4)` operand, with `cz` and `ccx`
      as controls, so the acceptance item is that **all four names and both
      controls** are in the suite. A test that only checks `x` does not
      establish the branch.
- [ ] The product-state call site at
      `flagquantum/simulation/statevector/product_state.py:480-486` is shown to
      be **reachable** with a matrix-bearing instruction, or the guard is added
      and the impossibility of reaching it is stated as the reason it cannot be
      proved by test. This session measured `product_state_calls=0` for 16-, 20-
      and 24-wire circuits with a matrix-bearing `z`, `h`, `s` and `y`. A guard
      that no test can reach is not evidence, and an unguarded site that can be
      reached is a defect, so one of the two has to be measured rather than
      assumed.
- [ ] The adjacency effect of Decision 2 is shown to be gone, with the exact
      six-operand table of "Problem" section 8 reproduced after the change:
      every one of the six operands is faithful at one gate and at two gates.
      The `identity` row is the control that shows the fixture is not simply
      asserting "folding changes the answer", and the `zeros(2, 2)` row is the
      control that shows a zero-norm program is no longer turned into `|0>` by
      an unrelated neighbouring gate.
- [ ] `remove_identity_gates` is shown to refuse a matrix-bearing `i`/`id` and a
      zero-angle `rx`/`ry`/`rz` by name, and `fq.plan` on such a program is shown
      to raise rather than to return a shorter plan. Section 13b of the probe
      already measures all **twelve** names of `_ROTATION_PARAM` at `theta = 0.0`
      as `planned n=0`, so at least the five arity-1 names and one arity-2 name
      have to be asserted after the change; the arity-2 half is what a test
      written from the arity-1 description would miss. The four rows of "Problem"
      section 4 are the fixture. The refusal is proven by deletion, not by the
      presence of the branch.
- [ ] The `_DIAGONAL_STATEVECTOR_GATES` dispatch in
      `flagquantum/runtime/dynamic/execution.py:115-116` is shown to apply the
      operand and not the diagonal, for `z`, `s`, `t` and `rz`. The six-row
      table of "Problem" section 5a is the fixture, and the `h` row is the
      control, because `h` already reads the operand.
- [ ] The crash at "Problem" section 5c is shown to be replaced by either a
      correct result or a named error, and the exact reproduction
      `DynamicCircuit(2).h(1).measure(1).gate('z',[0],matrix=X)` followed by
      `run_dynamic(..., shots=4)` is in the suite. A `RuntimeError` from
      `torch.multinomial` is the current outcome; `invalid multinomial
      distribution (sum of probabilities <= 0)` is the string to assert
      against.
- [ ] Each of the seven operands of "Problem" section 6 is classified by the
      new construction-time rule, with the expected and received shapes in the
      message. `(4,)`, `(1, 4)` and `(4, 1)` are refused (Decision 3), so the
      three rows that currently `RAN` become refusals and the four that
      currently fail at execution fail earlier. The message is asserted for one
      of each class, not for all seven.
- [ ] The `(2, 2)` non-unitary operand remains constructible, and
      `flagquantum.circuit.expectation` is shown to still work on the Hermitian,
      non-unitary operator that
      `tests/test_native_circuit.py::test_dense_expectation_preserves_complex_precision_and_gradients`
      passes. This is the item that distinguishes Decision 3 from the rejected
      alternative, and it must be measured on the real operator, whose
      `U^dagger U` is `0.2448` from the identity, and not on a fixture.
- [ ] The channel path is shown untouched: a circuit with `depolarizing`,
      `phase_damping` and a two-qubit channel is executed and its state is
      compared before and after the change. `instruction.matrix` is a Kraus
      4-tuple for each ("Problem" section 10), so a rule keyed on shape rather
      than on `is_channel` would break this and the test has to be able to see it.
- [ ] `tests/unit/test_folding.py::test_a_custom_operation_whose_matrix_is_not_unitary_is_refused`
      is updated for the `wide` half and its `shear` half is shown to be
      unchanged, with the failure kind before and after recorded. A test that
      merely stops raising is not evidence that the refusal moved to the right
      layer.
- [ ] `len(OPERATOR_SCHEMAS) == 35` and `IR_VERSION == "1.0"` are asserted after
      every change, and `get_operator_schema('any') is None` is asserted with
      them, so that a later registration cannot silently widen the table under
      the name of this change.
- [ ] `python tools/parity_matrix.py --check` reports
      `parity-matrix: contract valid and document current`, and
      `python tools/docs_source_of_truth.py --check` reports no stale generated
      document, and `python tools/check_capability_maturity.py --check` reports
      `capability maturity matrix passed`. Note that
      `docs_source_of_truth.py --check` exits `0` even when it prints a staleness
      line, so the item is the printed prose and not the exit status.
- [ ] The out-of-scope gap of "Problem" section 15 is confirmed to be **still
      present** after every change in this proposal, by re-running the three
      rows of that section. If a Decision below happens to fix it as a side
      effect, that is a finding and has to be reported rather than assumed
      either way — a program whose declared initial state is ignored by one
      entry point and honoured by another is exactly the kind of asymmetry this
      proposal exists to remove, and leaving its status unmeasured would be the
      same mistake one layer up.
- [ ] Repository owner approves. Not yet requested; see `## Status`.

## Non-goals

- **Honouring `Circuit(inputs=...)` in `fq.run`.** Measured, recorded and
  explicitly excluded in "Problem" section 15. It is a separate gap of the same
  class, it has its own fixtures, and folding it in here would put two unrelated
  defects in one proposal. It needs its own document.
- **A new registration function.** 067 proposes
  `flagquantum.operators.register_operator` and `OperatorRegistry`; this
  proposal proposes neither and does not name them in `## Public API`. Where the
  two meet is the capability record of Decision 7, which 067's registry is the
  natural container for.
- **A new root export.** The root surface stays at 36 names. Nothing here needs a
  new name, because every behaviour it changes is reachable through `Circuit`,
  `Circuit.any`, `Circuit.gate`, `fq.plan`, `fq.run` and
  `fq.experimental.dynamic.run_dynamic`, all of which already exist.
- **Making unitarity a precondition of `Circuit.any`.** Measured as
  incompatible with `flagquantum.circuit.expectation` in "Problem" section 11.
  The fold keeps its own unitarity test and the dense-expectation route keeps its
  exemption.
- **Rejecting non-unitary operands at execution.** Decision 2 declines to fold
  them; it does not refuse them. A single non-unitary instruction is applied
  exactly today ("Problem" section 8a) and continues to be.
- **Renaming or rewording `ExecutionError("planned execution failed")`.**
  Three suites pin the string and the wrapper is the correct one for an
  executor that fails after validation. Decision 5 moves failures out of it
  rather than changing it.
- **A gate-count or performance claim.** The fold's normalisation was found
  through `collapse_one_qubit_runs`, and 067's sibling work records that folding
  a `rz`/`sx` run was *undone* by `native_gate_legalization` at a 19% cost. This
  proposal changes when the fold fires and makes no claim that the change is
  faster or slower; that measurement belongs to whoever lands Decision 2.
- **A defined meaning for a flat operand.** Decision 3 refuses `(4,)`,
  `(1, 4)` and `(4, 1)` rather than blessing row-major `.reshape` semantics,
  because `.reshape` already accepts three layouts with no declared ordering and
  a fourth accepted layout would not be more meaningful than the three.
- **Photonic, control-system or provider-roster parity.** Unrelated rows of the
  parity matrix; this proposal does not touch them.
- **A `Pow`, `pow`, `conjugate` or symbolic modifier.** 067's non-goals already
  refuse these and this proposal adds nothing to that list.
