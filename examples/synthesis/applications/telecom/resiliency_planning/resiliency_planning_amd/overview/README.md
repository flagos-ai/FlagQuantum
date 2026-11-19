# CPU network resiliency planning

Run `python -m examples.synthesis.applications.telecom.resiliency_planning.resiliency_planning_amd.overview.main`, or execute `walkthrough.ipynb`.

All 29 original stages retain the five-node graph, seven edges, two demands, 24 assignment bits, twenty layers and forty-angle pi-linear schedule. The original active simulator is CPU. Its commented ROCm installation remains disabled. Native Circuit and structured result objects provide composition and sampling directly.

The original callback has a default of 1,000 shots, the final call requests 40,000 shots, and the archived source execution contains one actual objective invocation under `maxiter=1`. The native optimizer keeps this one-query budget, reports its stop and unconverged status, then draws an independent 40,000-outcome sample. All cost components and validity checks reuse the independently validated companion native Notebook. Both full complex state comparisons preserve absolute global phase and all 24 bits.

The source displays final probabilities using `NUM_SHOTS=2000`, despite drawing 40,000 outcomes. The report makes that denominator visible alongside actual normalized probabilities. It preserves the real sampled outcomes and plots valid assignments only when observed in the actual top 100. Native optimizer and resource metrics have their own declared basis.
