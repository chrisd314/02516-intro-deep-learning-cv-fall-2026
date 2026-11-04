# Project 3, Part 2: weakly supervised PH2 segmentation

This folder reserves a separate location for Part 2. Implementation is pending.

After the individual Part 1 fixes have been reviewed, Part 2 will receive its
own copies of the PH2 loader, selected architecture, splits, metrics and other
required helpers. Its code and experiment outputs will live in this folder.

Part 2 will use PH2 `_lesion` masks to simulate clicks and for evaluation.
Sparse training targets will identify unlabeled pixels explicitly. Point
generation, the dataloader, loss and training will each have their own review
checkpoint.
