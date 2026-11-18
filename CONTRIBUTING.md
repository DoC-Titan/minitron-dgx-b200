# Contributing

Start with the [experiment design](docs/EXPERIMENT.md) and [status](docs/STATUS.md).

- Keep changes to objectives, data, pruning rules and budgets explicit.
- Use a new run directory when a locked scientific protocol changes.
- Include commands, environment metadata and measured evidence with bug reports.
- Distinguish package/CPU checks from GPU execution.
- Preserve upstream notices and data provenance.

For a lightweight local check:

```bash
python DGX_B200_ONLINE/test_package.py
```

For the CPU math, checkpoint and rendering checks, install a CPU-capable PyTorch
and the helper dependencies, then run:

```bash
python DGX_B200_ONLINE/validate_local.py
```

Do not commit credentials, generated model weights, optimizer states, virtual
environments or synthetic fixtures as experiment results.
