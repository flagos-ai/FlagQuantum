# Workflow boundaries

This is a `single_device_fast_path` local workflow. `ansatz.py` owns native qfunc
models; `model.py` owns decoded classical costs; `workflow.py` owns optimization and
actual draws; `verification.py` owns independent Circuit comparisons;
`presentation.py` owns tables and figures. Only public FlagQuantum APIs are used.

Retain three layers and initial gamma[0,pi/2,pi], beta[pi,pi/2,0]. The source toy
MaxCut graph has edges(0,1),(0,2). Its QArray index0 is the decoded word's low bit.
Knapsack uses a in0..7,b in0..3,weights2/3,values3/5 and capacity12; slack in0..15.
The quantum models use all original widths and RX(beta) mixers on every bit.

A uniform state gives equal amplitudes before phase/mixer layers. Independent
Circuit references enumerate original classical diagonal phases and apply public
Circuit RX gates; they do not consume the native expression lowering or optimize
away global phase. Checks compare absolute complex amplitudes and separately sum
nonzero compiler-workspace probability. Resource counts precede Runtime compilation.

Native COBYLA is an explicit local optimizer adaptation. Its actual objective-query
cap preserves the source limits without claiming matching provider iteration or
trajectory semantics. Statistics sample1000 draws; the four Knapsack tables sample10
each, with separate explicit seeds. No classical optimum is fed to the optimizer,
postselected into observations or substituted for the last trace query. Exercise
solutions remain labeled and repeated execution stages are retained.

No provider credentials, remote submissions or hardware are needed. There is no
hardware, scalability, optimizer advantage, global optimality or speed claim.
