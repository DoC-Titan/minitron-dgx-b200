# DGX B200: upload-and-run notebook study

Upload/extract this **whole folder** into JupyterHub. Open `00_START_HERE.ipynb`,
choose your existing working Blackwell/PyTorch kernel, and run notebooks **00–08
in order**, one at a time. Each notebook starts isolated subprocesses with your
GPU UUID set before CUDA initialization. Training needs no external model API.

## What this runs

| Item | Default |
|---|---|
| Teacher | Existing pinned Qwen2.5-3B BASE at `/home/jovyan/work/minitron_dgx/models/qwen2.5-3b-pinned` |
| New run | `/home/jovyan/work/minitron_dgx/runs/b200_online_full_v1` |
| Structural reductions | 36→32 / 28 layers; FFN 11,008→9,600 / 8,192; hidden width unchanged |
| Pruning arms | Seeded random; squared-L2 magnitude; single-layer calibration sensitivity + magnitude FFN |
| Recovery | **Full-weight** student AdamW, FP32 parameters/states and BF16 autocast; teacher frozen BF16 |
| Teacher access | **Online, teacher and student GPU-resident together**; no logits disk cache |
| Conditions | 2 sizes × 3 criteria × 2 objectives (CE/KL) × 3 seeds = **36 runs** |
| Budget | 10 corpus epochs = **1,966,080 processed input tokens per run**, 1,920 updates |
| Data | 196,608 unique training input tokens, fixed calibration/validation/test and custom PIQA200 |
| Measurements | Immediate validation, recovery curves, final PPL/PIQA, parameters, memory, throughput, CPU/CUDA paired timings |
| CPU reference | Full tiny transformer train/prune/CE+KL path on CPU and CUDA, three seeds; matched-FP32 timing |

This is a **new expanded study**, not a continuation of the 18-run RTX study or
an exact resume of your earlier random student. The supplied prior magnitude cell
is retained in `PREVIOUS_CELL.md` as method context: it ranks arbitrary layers by
squared weight norm. New magnitude recipes use the same mathematical rule, with
stable tie-breaking. Sensitivity here is **individual-layer ablation**, not the
older RTX contiguous-window search. Neither guarantees the best joint cut.

## Relationship to your existing server work

The existing teacher is read from disk. Existing `random_32x9600_seed42` and
`magnitude_32x9600_recipe.json` are preserved. Those files alone do not identify the
earlier notebook's exact evaluation tensors, so the package bundles independently
verifiable prepared inputs and measures its own teacher baseline. **Do not compare
13.53/34.357 with new PPL as though the data were identical.**

Pinned source/data provenance and hashes: `UPSTREAM.json`. The dataset limitations
and original notices are retained under `licenses/`. Full teacher weights are not
in the upload ZIP. Your previous `models/download_teacher.py` creates the required
`DOWNLOAD_RECORD.json`; setup verifies its hashes and revision.

## Memory, runtime and storage

There is **no old 20 GB allocator cap**. The package uses only the configured GPU,
prints available resources, and runs an actual full-weight forward/backward/AdamW
smoke test before the main study. GPU memory is still finite: teacher + student +
gradients + optimizer + logits must fit. Full-weight training takes substantially
more memory than two inference models. Do not infer fit from weight size alone.

Allow roughly **300 GB free persistent disk** for 36 BF16 exports, active/reserve
optimizer checkpoints and temporary files; actual storage is measured by the server.
Host RAM must also hold checkpoint copies. Runtime is intentionally not promised.
Default training processes about **70.8 million input tokens** across all runs;
these are repeated epochs over a small corpus, not 70.8M unique tokens.

## Resume and checkpoints

- Rerun the training notebook to skip completed jobs and resume a partial job.
- An exclusive file lock prevents two workers using this run simultaneously.
- Checkpoints are written atomically with hashes and one previous generation.
- Create `STOP` inside the new run directory to pause after an optimizer update;
  the pause cell is in notebook 04. Remove that marker to resume deliberately.
- A sudden interruption can lose work since the last checkpoint (192 updates).
- Full models are exported and reloaded in BF16; a deterministic forward output
  is checked before completion is recorded. Do not delete checkpoints manually.
- After setup, source/config/data/software are locked. For different settings,
  choose a **new** `run_dir` and initialize again.
- An interruption after export rename but before metrics publication reuses the
  hash-verified export and compares its forward output with the final checkpoint.

## Deliverables

`reports/REPORT.md`, CSVs, compression/recovery/pruning plots, exact recipes,
per-run JSON, model exports, paired CPU/CUDA timings and `RESULTS.zip`. The report
refuses to claim completion if required measurements are missing. `RESULTS.zip`
contains compact evidence, **not** the large model checkpoints. Keep `models/`
on persistent storage for demonstrations.

`09_TENSORRT_SUPPORT.ipynb` records installed TensorRT-LLM support information.
It is conditional: the PyTorch pipeline does not fabricate a TensorRT engine or
claim unsupported status just because it is not installed. If your installed
release supports B200 and these pruned Qwen shapes, `tensorrt_optional.py` provides
conversion, engine building, repeated inference timing and paired PPL/PIQA checks.
Set the matching source/environment paths in notebook 09. This helper is explicitly
release-dependent and not locally TensorRT-tested; it rejects missing APIs rather
than silently falling back. This is the one backend-dependent setup step.

## Local validation boundary

See `LOCAL_VALIDATION.json` for tests actually run on the packaging machine.
CPU mathematical/structural tests are separate from the required B200 smoke test.
No B200 timing, recovery quality or GPU fit is claimed before you run the notebooks.
