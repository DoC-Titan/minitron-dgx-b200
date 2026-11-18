# Exact resumption context

Last supplied cell: load pinned teacher in BF16, sum squared parameter norms
per transformer layer, remove the four smallest; within retained layers, keep
the 9,600 largest coupled squared-L2 FFN channels; save
`runs/magnitude_32x9600_recipe.json`. That cell creates a **recipe**, not a trained
or exported magnitude-pruned model. It uses no calibration text and imposes a
20,000,000,000-byte PyTorch allocator limit.

This package applies that magnitude rule in a new run and adds a separate
calibration-sensitivity arm. The old recipe is not overwritten. Layer indices
are zero-based. Random and informed criteria are alternative student builds,
not two consecutive pruning stages on the same student.
