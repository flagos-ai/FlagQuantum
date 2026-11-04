# Native scalar coherent control

The unreleased `control(condition, body)` entry captures a zero-argument
lambda containing one scalar Hadamard or phase call. The condition must be
an initialized one-bit register disjoint from any target. A controlled
constant phase acts as a phase gate on the condition. Final gate counts
include the lowered controlled operation. The slice admits one level.
