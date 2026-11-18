# What is verified, and what is still pending?

| Component | Status | Evidence |
|---|---|---|
| Notebook syntax and schema | Passed locally | 10 notebooks |
| Balanced matrix / bundled input hashes | Passed locally | 3 packaging tests |
| KL direction, scaling and gradients | Passed on CPU | Runtime test |
| Physical shapes and save/reload | Passed on tiny CPU model | Runtime test |
| Exact-block data role separation | Passed locally | Runtime test |
| Full-weight optimizer recovery / frozen teacher | Passed on tiny CPU model | Runtime test |
| Report and chart generation | Passed with synthetic temporary fixture | Renderer test, not experiment evidence |
| B200 CUDA operation | Pending server execution | Notebook 00 |
| Simultaneous full-weight KD memory/update test | Pending server execution | Notebook 03 |
| 36 completed recovery conditions | Pending | Notebook 04 |
| DGX quality, timing and crossover findings | Pending | Notebooks 02, 04–07 |
| TensorRT release/build/runtime support | Conditional and unverified | Notebook 09 |

The source of truth for the packaging-machine checks is
[`LOCAL_VALIDATION.json`](../DGX_B200_ONLINE/LOCAL_VALIDATION.json).
Continuous integration checks package structure; a green CI badge would not prove
that B200 training has run or that it fits the assigned GPU allocation.

## Publishing future results

After the server finishes:

1. Run the completeness/report notebook.
2. Preserve the generated `state/protocol.json`, environment record and raw metrics.
3. Publish the compact `RESULTS.zip` as a release asset with its hash.
4. Add a results summary that links to those measurements and identifies hardware,
   precision, data scope and seeds.
5. Keep full checkpoint weights on persistent storage unless deliberately sharing
   them under the applicable Qwen license.

The existing RTX project is a separate completed study. Its measurements are not
substitutes for any pending row above.
