# Native two-dimensional quantum walk

The program keeps the original 32-by-32 toroidal grid, starting position
(16,16), and signed coin amplitudes (0.5,0.5,-0.5,-0.5). A step reflects the
coin about its uniform state, then moves x forward/backward for coin0/1 or y
forward/backward for coin2/3. Increment and decrement helpers express modular
arithmetic directly; users do not design ripple gates or name physical wires.

The native program accepts bounded static step counts and real position/coin
amplitudes. Defaults preserve the frozen original. Two nonzero test models
include positions31/0 and a coherent signed superposition straddling the grid
boundaries. The independent Circuit uses a direct uniform reflection and cyclic
permutation matrices, without invoking synthesis or its increment helpers.
Original/native/Circuit full complex states agree for steps0/1/2/3 without phase
alignment. Named x/y probabilities sum over the retained two-bit coin.

Dropping the coin hides its name; it does not reset, uncompute or reclaim its
entangled quantum environment. Only compiler work is required to return to zero.
Tests check every workspace amplitude as well as x/y marginals and actual
qubit/work/operation/depth/memory refusals. Default native20physical bits,
8work bits,775operations/depth605 versus original15physical bits,
198operations/depth186 is semantic acceptance with an explicit resource gap.
It is not complete Notebook workflow or optimization evidence.
