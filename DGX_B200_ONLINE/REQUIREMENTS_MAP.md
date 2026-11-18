# Page 1 and pages 20–21 evidence map

| Requirement | Notebook / resulting evidence |
|---|---|
| Open 1B–8B teacher, license and pin | 00; teacher receipt, environment.json, licenses/ |
| Two structural axes, physically fewer parameters | 01; exact recipes; runtime shape/reload tests |
| Magnitude/sensitivity vs simple baseline | 01; random/magnitude/sensitivity arms; pruning_maps.png |
| Teacher logit distillation | 03/04; frozen online teacher, forward KL at T=2 ×T² |
| Distillation ablation | 04; matched CE controls for every method/size/seed |
| Immediate vs recovered quality | 04/07; recovery, ranking.json, paired effects |
| PPL and downstream task quality | 02/04; common held-out blocks and custom PIQA200 |
| Memory, parameters, throughput | per-run JSON; benchmark JSON and CSV |
| Recovery against retraining tokens | recovery.png; unique vs repeated tokens distinguished |
| Compression-quality Pareto | compression_quality.png; pareto.json frontier membership |
| Exact structural groups | recipes/*.json, model pruning_recipe.json |
| CPU reduced train/prune/distil reference | 05; tiny_cpu.json and tiny_cuda.json |
| Actual NVIDIA GPU execution | 00/03 hardware operation and simultaneous residency smoke |
| Multiple sizes, warmups, repeats, median/IQR | 06; 32/128/512 prompts, 2 warmups, 5 timings |
| Environment metadata and multiple seeds | 00; environment.json; seeds 42/123/2026 |
| CPU/GPU scientific comparison | 06 paired PPL/PIQA subset; tiny matched FP32 quality |
| Separate workflow costs | phase log; load/preprocess/transfer/execution/decode/E2E fields |
| Observed beneficial GPU scale | report per-variant minimum observed win; no unsampled claim |
| ModelOpt/NeMo where appropriate | custom PyTorch slicing justified by direct shape control; not used |
| TensorRT-LLM if supported | 09 support audit; conditional external build/benchmark pending if supported |

No inference is made about CPU/GPU equivalence from parameter identity alone.
Real-model CPU FP32 vs CUDA BF16 ratios combine precision and hardware. Tiny
matched-FP32 uses identical weights on both devices and measures quality too.
Subword Qwen perplexity and byte-level tiny perplexity are not comparable.
