# Project 3: image segmentation

The assignment parts have separate locations:

```text
Group_Proj3/
  part1/                     # Existing fully supervised PH2/DRIVE work
    train.py
    predict.py
    measure.py
    plot_results.py
    modelrun.sh
    lib/
    checkpoints/
    plots/
    README.md
  part2/                     # Weak supervision: implementation pending
    README.md
```

See [Part 1](part1/README.md) for the relocated code and
[Part 2](part2/README.md) for its current status.

The first checkpoint separated the folders and updated the HPC script's project
path to the `part1` subfolder. Existing checkpoints and plots were moved with
their original filenames and contents. Subsequent individual corrections are
documented in the Part 1 README's review status.

Run the current Part 1 commands from `Group_Proj3/part1`, because their output
paths remain relative to the working directory. Dataset locations remain
unchanged; the loaders still use `/dtu/datasets1/02516` on HPC.

Part 1 fixes, local verification, technical notes and corrected historical
checkpoint evaluations are complete. See its README for the remaining fresh
HPC training sweep and the limitations of historical training provenance.
Part 2 will later receive its own copies of the approved modules it needs,
with no imports from Part 1. Its implementation has not started.
