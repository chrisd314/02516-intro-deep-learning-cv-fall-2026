"""Fixed, reproducible PH2 clicks at the model's resized image resolution."""

import torch
import torch.nn.functional as F


STRATEGIES = ("uniform", "interior", "boundary")


def image_seed(seed, image_id):
    """Use the case ID, never Python's randomized hash or dataset order."""
    return (int(seed) + int(image_id.upper().removeprefix("IMD")) * 1000003) % (2**63 - 1)


def sample_points(target, positive=10, negative=10, seed=42, strategy="uniform", margin=3):
    """Return int64 [1,H,W] labels (-1 unknown, 0 background, 1 lesion) and counts.

    Interior/boundary sampling uses a square neighborhood of radius ``margin``.
    If the preferred region is too small, fill from other pixels of the same
    class. Never duplicate pixels or change their class to meet a click budget.
    """
    if target.ndim != 3 or target.shape[0] != 1 or target.numel() == 0:
        raise ValueError("target must have nonempty shape [1,H,W]")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("target must be binary")
    if positive < 0 or negative < 0 or seed < 0 or margin < 1:
        raise ValueError("Click counts/seed must be nonnegative and margin positive")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    mask = target.detach().cpu().bool()
    labels = torch.full(mask.shape, -1, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    details = {}
    for name, value, requested in (("positive", 1, positive), ("negative", 0, negative)):
        region = mask if value else ~mask
        if strategy == "uniform":
            preferred = region
        else:
            # Pool padding ignores pixels outside the image, so the image edge
            # alone does not count as a lesion/background boundary.
            opposite = F.max_pool2d((~region).float().unsqueeze(0),
                                    2 * margin + 1, stride=1, padding=margin)[0] > 0
            interior = region & ~opposite
            preferred = interior if strategy == "interior" else region & ~interior
        preferred_ids = preferred.flatten().nonzero().flatten()
        other_ids = (region & ~preferred).flatten().nonzero().flatten()
        # Full permutations make smaller budgets prefixes of larger budgets
        # for both classes, even when the requested positive count changes.
        preferred_ids = preferred_ids[torch.randperm(len(preferred_ids), generator=generator)]
        other_ids = other_ids[torch.randperm(len(other_ids), generator=generator)]
        ordered = torch.cat((preferred_ids, other_ids))
        chosen = ordered[:min(requested, len(ordered))]
        labels.view(-1)[chosen] = value
        coordinates = [[int(index) // mask.shape[2], int(index) % mask.shape[2]]
                       for index in chosen]
        details[name] = {
            "requested": requested, "sampled": len(chosen), "available": len(ordered),
            "preferred_available": len(preferred_ids),
            "fallback": max(0, len(chosen) - len(preferred_ids)),
            "coordinates_yx": coordinates,
        }
    return labels, details
