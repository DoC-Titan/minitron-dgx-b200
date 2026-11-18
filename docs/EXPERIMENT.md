# Experiment design

## One budget. Three selection rules.

Each rule produces a physically smaller student at a predeclared size. The rules
are alternatives, not successive pruning stages on the same model.

| Rule | Which layers disappear? | Which FFN channels survive? |
|---|---|---|
| Random | Seeded arbitrary subset | Seeded random coupled channels |
| Magnitude | Smallest total squared parameter norms | Largest coupled squared-L2 channel scores |
| Sensitivity | Lowest single-layer calibration-ablation NLL damage | Largest coupled squared-L2 channel scores |

All indices refer to the original teacher. The FFN operation slices gate/up rows
and the corresponding down-projection columns together. Hidden width remains
2,048; this is depth and **FFN-width** pruning, not residual-width pruning.

The sensitivity arm ranks individual layer ablations. It does not search every
joint subset or guarantee the smallest damage after combining depth and FFN cuts.
The FFN magnitude map is descriptive weight scoring, not causal channel ablation.

## Recovery matrix

| Per compression level | CE | KL | Total |
|---|---:|---:|---:|
| Random, three seeds | 3 | 3 | 6 |
| Magnitude, three seeds | 3 | 3 | 6 |
| Sensitivity, three seeds | 3 | 3 | 6 |
| **Total per size** | **9** | **9** | **18** |

Two compression levels give **36 runs**. CE and KL share the same pruning recipe,
training data, seed-controlled order, optimizer settings and processed-token budget.

- **CE:** negative log-probability of the actual next token.
- **KL:** forward `KL(teacher || student)` over the full vocabulary; temperature
  2 with the conventional temperature-squared factor. No hard-label CE term.
- Both start from retained pretrained teacher weights. CE has no teacher-output
  supervision during recovery; it is not training a student from random weights.
- The teacher remains frozen. All retained student weights are trainable.

## Fixed data roles

| Partition | Purpose |
|---|---|
| 32 calibration blocks | Choose sensitivity scores |
| 1,536 training blocks × 128 tokens | Recovery; repeated for ten epochs |
| 64 validation blocks | Immediate damage and recovery curves |
| 256 test blocks | Fixed final language-model evaluation |
| Custom fixed PIQA200 | Downstream physical-commonsense accuracy |

There are 127 within-block next-token targets per 128-token input. Input-token
counts and scored-target counts should not be confused. Checkpoints are reported
at a fixed budget; the test set is not used to select a winning checkpoint.

Exact-block overlap checks are narrower than semantic deduplication or
pretraining-contamination guarantees. These are reused disclosed inputs, not a
new unseen benchmark. See the retained upstream data card for provenance.

## Interpret the results carefully

1. Compare paired KL/CE outcomes within a rule, size and seed.
2. Compare selection rules at matched parameter counts.
3. Inspect immediate versus final rankings on the same validation partition.
4. Use test PPL and PIQA as complementary measures, not interchangeable quality scores.
5. Report means and standard deviations across three seeds without claiming a
   universal winner or statistical superiority from a small pilot.
6. Compare observed Pareto points without extrapolating an unmeasured frontier.

No outcome is prewritten: a random rule or a CE control may outperform the more
elaborate method. Negative findings are valid results.
