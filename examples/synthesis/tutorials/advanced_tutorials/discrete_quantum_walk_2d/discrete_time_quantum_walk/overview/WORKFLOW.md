# Original stage mapping

| Original code cell | Native code cell | Stage |
| --- | --- | --- |
| 2 | 2 | Local native setup |
| 3 | 4 | Original one-dimensional size |
| 4 | 6 | Native ring preparation and helpers |
| 5 | 8 | One-step ring synthesis and full-state check |
| 6 | 10 | One-step actual observations |
| 7 | 12 | Twenty-step actual observations and ring plot |
| 9 | 14 | Original two-dimensional size |
| 10 | 16 | Reused native torus preparation and helpers |
| 11 | 18 | Twenty-step torus synthesis and full-state check |
| 12 | 20 | Actual observations and probability heatmap |

The native Notebook preserves every original size, initial condition, step count
and observation count. Source runtime step bindings become checked synthesis-time
bindings; separate ring builds make this difference explicit. Local Circuit
drawings replace the hosted IDE. The full retained coin state is compared before
position marginalization, with compiler storage checked separately. No gate
resource parity or classical/quantum performance advantage is claimed.
