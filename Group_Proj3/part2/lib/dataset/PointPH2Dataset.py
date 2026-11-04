"""A training-only dataset returning images and cached sparse labels."""

from torch.utils.data import Dataset

from lib.points import image_seed, sample_points


class PointPH2Dataset(Dataset):
    def __init__(self, base, positive=10, negative=10, seed=42,
                 strategy="uniform", margin=3):
        if base.split != "train":
            raise ValueError("Generate simulated annotations only for the training split")
        self.base = base
        self.sample_ids = list(base.sample_ids)
        self.labels = []
        self.annotations = {}
        for index, image_id in enumerate(self.sample_ids):
            # Full masks are used once to simulate annotations, then discarded.
            case_seed = image_seed(seed, image_id)
            labels, details = sample_points(base[index]["target"], positive, negative,
                                            case_seed, strategy, margin)
            self.labels.append(labels)
            self.annotations[image_id] = {"seed": case_seed, **details}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        image, _ = self.base.load_image(index)
        # The training batch has no complete target or full-mask valid region.
        return {"image": image, "points": self.labels[index].clone(),
                "image_id": self.sample_ids[index]}
