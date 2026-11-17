# Sampled physical operation counts

This command implements locked K-0032 and K-0033. The source command
and test blobs are `78ef4ecbe7bd58732ae73199f39a97960d5a65dc` and
`2ee5b835780739949feb949aeffae4542b2d76d8`. The numerical dependency
is the input-basis transition and counting portion of `Sim::apply`
at blob `adbacdb20e3351fb826c9e244085eb81e3a18255`.

```console
python -m flagquantum.ecosystem.kmx_sampled_counts --in circuit.kmb --shots 65 --groups
```

The command accepts KMX and KMB files or stdin, and writes to stdout
or the requested path. `--shots 0` prints zero counts with undefined
retention percentages. `--init r0=5,r1<0x6,r2=*` initializes mixed
quantum and classical registers in their physical member order, with
the first member as the low bit. Fixed input, a bound on uniform
random input and full-width random input are supported.
Undeclared initialization registers are rejected with a diagnostic.

The format reader and report stay in Compiler. The command composes
them with Simulation through the Ecosystem boundary. The simulator
packs at most 64 computational-basis trajectories and counts each
record using its active guard mask before applying the operation.
Scope guards are captured on entry. HMR produces an unbiased
X-basis result, clears its target and retains the conditional sign.
Reset clears its target using the same branch law without recording
the result. Diagonal gates and HMR signs accumulate in the exact
128-bit phase ring. These input-basis trajectories are the source
semantics for this command. The native random stream is independent
of the source's externally seeded stream.

Formal scope and tag rows always retain their static multiplicity.
The grouped total excludes debug, tag and scope records. In sampled
mode the metadata subtotal counts NEG. Counts and retention are
rendered with the original integer truncation and padding. They
measure branch-dependent physical records separately from static
opcode counts and modeled CCZ states.

The original all-operation test passed. The native command exercises
its file route with all opcodes in KMX and KMB form. Six deterministic
source reports match byte for byte, including fixed and bounded
register initialization and a 65-shot tail. Complementary HMR guards
retain a constant total across batches. A source simulation probe
records 64 shot outcomes, all four two-measurement histories, output
basis states and exact phases. Native packed trajectories reproduce
all 64 rows when fed the same outcomes. The existing dynamic
`circuit.append(op)` execution also agrees on every complex amplitude
in each observed history, with the spectator work bit at zero.
