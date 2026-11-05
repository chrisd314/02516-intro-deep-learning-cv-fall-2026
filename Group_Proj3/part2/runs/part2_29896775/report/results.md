# PH2 fully versus weakly supervised results

Completed records included: 8 of 8 planned.
Configuration: unet, size 128, 20 epochs, seed 42.

All losses in metrics_summary.csv are dense evaluation BCE, including for sparse models.
Sparse training loss is stored separately in each epoch history and uses annotated pixels only.

| Experiment | Test Dice | Test IoU | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| fully supervised | 0.878304 | 0.783014 | 0.933716 | 0.834958 | 0.973366 |
| uniform 1+1 | 0.777505 | 0.635999 | 0.880733 | 0.727428 | 0.942283 |
| uniform 5+5 | 0.835098 | 0.716882 | 0.905210 | 0.837835 | 0.932261 |
| uniform 10+10 | 0.853663 | 0.744688 | 0.919668 | 0.817921 | 0.960518 |
| uniform 20+20 | 0.831354 | 0.711382 | 0.903721 | 0.828375 | 0.933972 |
| uniform 50+50 | 0.866360 | 0.764229 | 0.921173 | 0.891916 | 0.932920 |
| interior 10+10 | 0.852798 | 0.743372 | 0.915948 | 0.849894 | 0.942469 |
| boundary 10+10 | 0.670281 | 0.504077 | 0.794493 | 0.729168 | 0.820720 |

The smallest evaluated uniform budget reaching 95% of full validation Dice is uniform 5+5.

The approach criterion is 0.95 times the full reference's validation Dice,
specified before comparison. Test scores do not select hyperparameters or checkpoints.

Full validation masks are used for evaluation/checkpoint selection in both methods.
The sparse annotation budget applies to training images only. Full training masks
generate clicks once and are otherwise used only for final evaluation/visualization.

Clicks are fixed across epochs, unique and sampled at the resized image resolution.
Interior/boundary sampling may fall back to other same-class pixels; exact fallback
counts and coordinates are saved in .points.json and annotation_summary.

These are single-training-seed and single-click-seed comparisons, without confidence
intervals. Low-resolution masks, sampling from perfect GT and fully labeled validation
limit claims about real user annotations. Scores alone do not establish causal reasons.

Representative images use the first two fixed IDs, not the best test predictions.
Training curves, click-count plots, strategy comparisons and all split scores are separate artifacts.
