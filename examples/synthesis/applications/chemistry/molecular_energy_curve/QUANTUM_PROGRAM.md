# Retained molecular quantum ansatz

Complete retained one-qubit molecular quantum ansatz. Two nonzero angles, both width/depth objectives, exact original/native/independent Circuit complex amplitudes, clean workspace, accepted one-instruction budget and insufficient-memory refusal are checked. Complete chemistry Notebook data/energy/optimization workflow remains pending.

The retained preparation helper initializes |1>; the original one-term multi_suzuki_trotter is expressed by existing native suzuki_trotter(-0.5*Y,theta,1,1). Independent X then RY(-theta) preserves the generator sign and absolute phase. This does not add a general multi_suzuki_trotter API. Both upstream paths happen to have byte-identical quantum sources but remain separate inventory cases. No production Compiler/API/budget changes; native user examples contain no source importer.

Full complex128 amplitudes use absolute tolerance 1e-12 without phase alignment.
The 0.37 binding uses one qubit, one u3 instruction and depth one; the -0.81
binding uses one u3 plus one RZ and depth two. Both use zero workspace, under
both synthesis objectives. These are static IR measurements. The initial test
incorrectly expected a one-instruction ceiling to refuse for 0.37; measured
resources and execution establish that this ceiling is feasible for that binding.
The insufficient-memory refusal remains tested.

[README](README.md) gives native and Circuit commands. Notebook workflows require separate acceptance.
