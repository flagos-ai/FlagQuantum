# Literal source workflow and semantic boundaries

| Source code cell (all commented) | Native counterpart |
| --- | --- |
| 3: imports and backend | Public FlagQuantum CPU execution; no foreign SDK |
| 4: QFT and constant adder | Checked quantum functions with precomputed finite real angle arrays |
| 5: reflection, block encoding, projectors and QSVT | Literal target placement, half-turn conversion and positional box mapping |
| 7: size 6, degree 3, four phases and compilation costs | Same logical workload; actual CX/U3 counts, depth, width and monotonic local durations |

The entire source is commented. This is a native reconstruction, not execution
of Tket, certification of its constructor syntax, a reproduction of its
optimization passes or a comparison of compilation times. The scalar
`control_state=0` is interpreted as one false control.

Angles require care: Tket Rz consumes half turns, while the adder source also
multiplies its expression by pi. Native RZ therefore receives
`(constant % 2**(j+1)) * pi**2 / 2**j` radians. These literal Fourier rotations
are not modular addition. Projector RZ angles likewise gain a pi factor.
The source reflection applies X/H around data[0] while its zero-controlled X
targets data[5]. It is not a diagonal zero-state reflection. An independent
64-column operator comparison verifies the exact H/phase/H identity used here.

CircBox orders the internal named registers block before data. The source
passes positional operands data before block. Each U/U-dagger therefore binds
its internal block to outer data[0:3], and its internal data to
[outer data[3:6], outer block]. Projectors still address outer block.
The native code makes that permutation explicit rather than silently repairing
it. Named output wire order is MSB-first [block,data,aux]; inside U the physical
select, deletion and projection wires are 8, 7 and 6, and internal data is
[5,4,3,2,1,0]. QNum indexing itself is LSB-first.

Native and CX/U3 complete complex states are compared on zero, nonzero boundary
and complex coherent inputs, and a second nonzero four-angle binding. Absolute
phase is retained. This is not exhaustive verification of all 1024 full-program
columns. The logical block and auxiliary bits remain outputs; there is no
compiler work. The independent reference's largest dense complex128 matrix is
256 by 256 (one MiB), solely for correctness inspection. Synthesis uses no
dense matrices.

The local policy explicitly adds ten physical bits, zero compiler work,
20,000 native instructions, 200,000 added target operations and 64 MiB of
execution memory. The source supplies no such budgets. Memory policy is not a
process-RSS bound. Native and conversion budget refusals are tested.
Caller-owned exact RX/RZ source expansion runs before generic quarter-turn
simplification, which can discard scalar phase. S/SDG conversion rules and
all-column RX/RZ probes preserve absolute phase. No shared conversion table
is modified.

This workflow is single_device_fast_path. Counts and durations are local
observations, without optimality, speedup, hardware or release claims. Source
Notebook outputs remain empty; executed Jupyter artifacts live outside examples.
