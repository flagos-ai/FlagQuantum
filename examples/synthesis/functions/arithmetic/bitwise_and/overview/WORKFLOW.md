# Unsigned bitwise-AND workflow

The complete reviewed Notebook synthesizes and samples two programs: register
AND with a=4 and b=5, yielding 4; then literal AND with a=5 and mask 3, yielding
1. This native workflow reuses both corresponding FlagQuantum qfunc examples
and runs their complete host synthesis/execution/sampling steps. The user
programs retain a & b and 3 & a expressions without gate design or bit decoding.

1000 seeded local draws from executed named probabilities replace each cloud
sampling call. The notebook display becomes plain Python count dictionaries:
(4,5,4) and (5,1). Independent Circuit comparisons for both programs check
absolute complex amplitudes, logical order and clean workspace. Shared tests
also check coherent word operations and zero extension under both schedules.
Signed OR/XOR Notebook workflows remain pending; their first programs require
signed native allocation and in-place XOR beyond this unsigned profile.
