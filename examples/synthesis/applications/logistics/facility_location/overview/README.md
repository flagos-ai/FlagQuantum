# Native facility-location optimization

Run `python -m examples.synthesis.applications.logistics.facility_location.overview.main`, or execute `walkthrough.ipynb` from the repository root.

The original three facilities and four customers use twelve decision bits. Every customer has one supplier and exactly two facilities serve customers. The nonlinear squared penalty counts empty facilities; it is not replaced with a linear approximation. All five QAOA layers and the original fifty-iteration limit remain fixed.

Public native exact means and Nelder-Mead replace proprietary shot-based optimization. The source omits its quantile value; the current reviewed SDK defaults to one, but its version does not establish the historical Notebook environment. The native mean objective is explicit. Each angle tuple is synthesized anew. Actual finite optimized and zero-angle uniform samples retain cost, penalty and feasibility; an exhaustive constrained classical solve runs after sampling and never repairs observations.

The complete seventeen-stage workflow includes the original network, model display, native circuit/resources, trace, sampled top five, uniform comparison histogram, observed table and graph, and separate classical table and graph. See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json). Parent native quantum and Circuit examples remain frozen.

The unchanged nonlinear quantum program uses twelve decision bits and five
clean temporary work bits, seventeen physical bits in total. The synthesis
budget admits these work bits; it does not count them as decisions or slack.
