# Numerical OnEach composition

Simulation owns `classical_on_each.on_each_classical_value`, adapted from
unmodified Qualtran v0.7.0 `OnEach.build_composite_bloq` and its numerical graph
execution at `bcb17bb25360ed68ce3bc084645db3e929799389`.

The metadata owner supplies an already-bound single-q THRU child evaluator and
word validation, encoding and decoding callbacks. Child-signature construction,
target-dtype assertions and unsupported non-THRU bindings are not reimplemented
as another graph hierarchy. The kernel reuses existing Split/Join, bit validators,
integer-array output collection and phased-output commit.

Input validation precedes the source Expr-only iteration-width rejection. The
separate Split dtype-symbolic guard then uses the bound register width. Iteration
count and register width remain distinct: `register_width` can identify the fixed
one-bit target even when the loop count has a different Python type or symbolic
protocol. Metadata array allocation and index checks happen before numerical
encoding, matching concrete builder failures. A graph owner must validate other
constructor/signature mismatches before supplying the bound evaluator.

For a nonempty split register, encoding is followed by exact array-shape and bit
validation. Children consume bits MSB first. Their returned values are validated
and retained as objects in indexed assignments; their resolved phases pass through
the shared ordered commit helper. No eager integer conversion is performed on a
child result. Join collects the bit register into the original uint8 container,
then calls `tolist` and the bound word decoder. Final word validation is retained.
In particular, the source bit domain is not replaced with stricter QUInt(1)
scalar rules: floating zero/one child results can remain floats until collection.

A zero-bit register is a special source topology: the independent Join executes
before Split. Standard unsigned decoding raises `ValueError` on the empty bit
string; signed decoding raises `IndexError`. If a valid custom codec accepts the
empty Join, Split still runs afterwards. Both the failure ordering and successful
custom-codec ordering are tested against the actual source graph. Zero loop
iterations with a fixed nonempty QBit register are a different case and preserve
the bit. Neither case is silently replaced with the compiler's empty-gate no-op.

Run the native user example:

```sh
python -m examples.algorithms.classical_on_each_values
```

It composes X then Z on each of three bits, or their exact per-bit reverse, using
`circuit.append(op)` and the existing `on_each_ir` emitter. Noncontiguous wires
preserve a middle spectator. Full complex amplitudes, the complete original
tensor and inverse restoration agree. There are six emitted Clifford instructions,
zero T, zero measurements and zero work wires; four declared wires include the
spectator, while the source model has three peak logical wires. There is one
unitary branch in this example, not a measured branch ensemble.

Reference tests use original full numerical graphs with opaque, unsigned, signed
and ones-complement targets, raw value diagnostics and native codecs. They retain
NumPy/Sympy width behavior, expression versus protocol distinctions, and the
fixed QBit target exception to count-width assumptions. A local source contract
child demonstrates that float bit results remain floats in source assignments;
a replacement collector proves native conversion happens only at Join. Original
graph traversal checks every X/Z primitive and wire mapping in source call order.
Both QEC profiles and peak width are compared separately from emitted counts.

The source reference is unchanged. Complete OnEach constructor/diagram methods,
arbitrary non-THRU child bindings, global graph topology and graph-owned exception
translation remain outside this numerical composition slice.
