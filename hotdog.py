"""
hotdog.py -- helper code for the Hotdog / Not-Hotdog project (DTU 02516).

All the reusable / slow code lives here so the notebook stays short and readable.

Use it from the notebook like this:

    import hotdog
    # If you EDIT this file, run this line to pick up the changes
    # (a plain `import` won't re-read a file Python already loaded):
    import importlib; importlib.reload(hotdog)

Contents:
    1. Data      -- loading, train/val split, augmentation
    2. Models    -- a small CNN you can configure, and ResNet18 for transfer learning
    3. Training  -- one train loop used by every experiment
    4. Results   -- save/load runs so you never have to retrain for a figure
    5. Saliency  -- vanilla saliency + SmoothGrad
    6. Plotting  -- small helpers for the report figures
"""

import os
import glob
import json
import time
import random
import functools

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset


# Where results get written. Created automatically on first use.
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
FIGURE_DIR = "figures"

# The class names, in the order the labels are numbered.
# (Folders are sorted alphabetically, so hotdog=0 and nothotdog=1.)
CLASS_NAMES = ["hotdog", "not hotdog"]

# Standard ImageNet normalisation values (mean and std of each RGB channel).
# We use these everywhere. They are *required* for the pretrained ResNet
# (it was trained on images normalised this way), and using the same numbers
# for our own CNN keeps the code simple with no downside.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=0):
    """Make a run repeatable: same weights init, same shuffling, same augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Use the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. DATA
# =============================================================================

@functools.lru_cache(maxsize=None)
def _load_split(data_path, split, size):
    """
    Read every JPEG of one split ('train' or 'test') into memory, ONCE.

    Decoding JPEGs is the slowest part of an epoch (~19s per epoch here), and
    it produces the exact same pixels every time. So we do it a single time and
    keep the result in RAM as uint8 (1 byte per colour per pixel, ~100 MB for
    the whole training set). Every later epoch then reads straight from memory.

    @lru_cache means the second call with the same arguments returns the cached
    result instantly instead of re-reading the disk.

    Returns:
        images: uint8 tensor of shape (N, 3, size, size)
        labels: int64 tensor of shape (N,)   -- 0 = hotdog, 1 = nothotdog
    """
    folder = os.path.join(data_path, split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Could not find '{folder}'. Check the data_path argument -- it should "
            f"point at the folder containing 'train' and 'test'."
        )

    # Sub-folder names are the class names; sorting makes the labels deterministic.
    class_names = sorted(
        os.path.basename(d) for d in glob.glob(folder + "/*") if os.path.isdir(d)
    )
    name_to_label = {name: i for i, name in enumerate(class_names)}

    paths = sorted(glob.glob(folder + "/*/*.jpg"))
    images = torch.empty(len(paths), 3, size, size, dtype=torch.uint8)
    labels = torch.empty(len(paths), dtype=torch.long)

    # Resize once, here, to the size we train at.
    resize = T.Resize((size, size), antialias=True)

    for i, path in enumerate(tqdm(paths, desc=f"loading {split}", leave=False)):
        # .convert('RGB') matters: a few ImageNet images are greyscale or CMYK,
        # and without this they'd come out with 1 or 4 channels and crash training.
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC -> CHW
        images[i] = resize(img)
        labels[i] = name_to_label[os.path.basename(os.path.dirname(path))]

    return images, labels


class HotdogDataset(Dataset):
    """
    Wraps the in-memory images and applies a transform to each one.

    Two datasets can share the same `images` tensor while applying *different*
    transforms -- that is how we give the training set augmentation while the
    validation set stays clean.
    """

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def _make_transform(size, augment):
    """
    Build the preprocessing pipeline.

    Both pipelines end the same way: convert uint8 [0,255] -> float [0,1],
    then normalise. Augmentation only adds random steps at the front, and only
    ever for training -- validation and test must stay fixed, otherwise the
    numbers you compare would be noisy for the wrong reason.
    """
    steps = []
    if augment:
        steps += [
            # Left-right flip: a mirrored hotdog is still a hotdog.
            T.RandomHorizontalFlip(),
            # Random zoom/crop: teaches the net that position and scale vary.
            T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            # Small colour wobble: robustness to lighting differences.
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] -> float [0,1]
        T.Normalize(MEAN, STD),
    ]
    return T.Compose(steps)


def get_loaders(data_path="hotdog_nothotdog", size=128, batch_size=64,
                augment=False, val_fraction=0.2, seed=0, num_workers=0):
    """
    Build the three DataLoaders we need.

    Why a validation set? The test set is only allowed to be used ONCE, at the
    very end, to report the final number. Every decision you make along the way
    (which architecture, which optimizer, whether augmentation helps) has to be
    made on data the model never trained on but that isn't the test set --
    that's the validation set. Choosing your model by test accuracy makes your
    reported test accuracy optimistic and is the classic way to lose marks.

    The split is seeded, so train and validation stay the same across every
    experiment and the comparisons are fair.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_images, train_labels = _load_split(data_path, "train", size)
    test_images, test_labels = _load_split(data_path, "test", size)

    # Shuffle the training indices once, then cut off the last `val_fraction`.
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_labels), generator=generator)
    n_val = int(len(train_labels) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    # Same underlying images, different transforms: augment train, not val.
    train_full = HotdogDataset(train_images, train_labels, _make_transform(size, augment))
    val_full = HotdogDataset(train_images, train_labels, _make_transform(size, False))
    test_set = HotdogDataset(test_images, test_labels, _make_transform(size, False))

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    # shuffle=True for training only -- the order shouldn't be a signal the net
    # can learn from. Evaluation order doesn't matter, so we leave it fixed.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


def denormalize(x):
    """
    Undo the normalisation so an image can be displayed.

    Normalised tensors contain negative numbers, which matplotlib would clip
    into nonsense colours. This puts them back into the [0,1] range.
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1)


# =============================================================================
# 2. MODELS
# =============================================================================

def make_cnn(n_blocks=4, width=16, batch_norm=True, dropout=0.0):
    """
    A small VGG-style CNN, with knobs so we can run the experiments the
    project asks for by changing arguments instead of rewriting the model.

    One "block" is:  Conv 3x3  ->  (BatchNorm)  ->  ReLU  ->  MaxPool 2x2

    The MaxPool halves the image each block (128 -> 64 -> 32 -> 16 -> 8),
    while the number of channels doubles, so the network trades spatial detail
    for a richer description of *what* is in the image. This "get smaller and
    deeper" shape is the standard CNN design.

    Args:
        n_blocks:   how deep the network is
        width:      channels in the first block (doubles every block)
        batch_norm: include BatchNorm layers (one of the report questions)
        dropout:    dropout before the final layer, 0.0 = off. Fights overfitting.
    """
    layers = []
    c_in = 3  # RGB input
    c_out = width
    for _ in range(n_blocks):
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
        if batch_norm:
            # BatchNorm renormalises each channel per batch. It usually lets you
            # train faster and with a higher learning rate.
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        c_in = c_out
        c_out = c_out * 2

    # Average each channel down to a single number, then classify.
    # This has far fewer parameters than flattening, which matters a lot with
    # only ~1600 training images.
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(c_in, 2))  # 2 outputs = 2 classes

    return nn.Sequential(*layers)


def make_resnet18(pretrained=True, freeze=False):
    """
    ResNet18 for transfer learning.

    The idea: a network already trained on ImageNet's 1.2M images has learned
    generic visual features (edges, textures, shapes). Reusing those beats
    learning from scratch on our ~1600 images.

    Args:
        pretrained: start from the ImageNet weights (True) or from scratch (False).
                    Training the identical architecture with pretrained=False is
                    the fair comparison that shows how much transfer learning helps.
        freeze:     if True, only the new final layer is trained and the rest of
                    the network is used as a fixed feature extractor. Much faster,
                    and less prone to overfitting on a small dataset.
                    If False, the whole network is fine-tuned.
    """
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # ResNet18 ends in a 1000-class layer (ImageNet). Swap it for a 2-class one.
    # A newly created layer always has requires_grad=True, so this one trains
    # even when everything else is frozen.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def count_parameters(model):
    """Number of trainable parameters -- handy for the report's model table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 3. TRAINING
# =============================================================================

def make_optimizer(model, name, lr, weight_decay=0.0):
    """
    Build one of the optimizers the project asks you to compare.

    Only parameters with requires_grad=True are passed in, so this also works
    for a frozen ResNet.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Use 'sgd', 'momentum' or 'adam'.")


@torch.no_grad()  # no gradients needed when evaluating -> faster, less memory
def evaluate(model, loader, device=None):
    """
    Run the model over a whole dataset without training on it.

    Returns a dict with the loss, the accuracy, and the raw predictions
    (needed later for the error analysis).
    """
    device = device or get_device()
    model.to(device).eval()  # eval() matters: it switches BatchNorm/Dropout
                             # into inference mode. Forgetting it is a very
                             # common bug that makes val accuracy look wrong.

    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return {
        "loss": total_loss / len(labels),
        "acc": (preds == labels).float().mean().item(),
        "preds": preds,
        "labels": labels,
        "probs": torch.cat(all_probs),
    }


def train_model(model, train_loader, val_loader, name, epochs=10,
                optimizer="adam", lr=1e-3, weight_decay=0.0,
                device=None, seed=0, verbose=True):
    """
    Train a model and record everything the report needs.

    Every experiment in this project is this same function with different
    arguments -- that is the whole point of putting it here.

    After each epoch we check validation accuracy and keep a copy of the best
    weights ("early stopping by checkpoint"). If the model starts overfitting
    later on, we still end up with its best version.

    Results are saved to results/<name>.json and checkpoints/<name>.pt, so you
    never have to retrain a model just to redraw a figure.

    Returns:
        history dict (also written to disk)
    """
    set_seed(seed)  # same starting point for every run -> a fair comparison
    device = device or get_device()
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, weight_decay)

    history = {
        "name": name, "epochs": epochs, "optimizer": optimizer, "lr": lr,
        "weight_decay": weight_decay, "n_params": count_parameters(model),
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()  # training mode: BatchNorm updates its statistics,
                       # Dropout is active
        running_loss, correct, seen = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)

            # --- the four lines that are the whole of training ---
            opt.zero_grad()                                  # clear old gradients
            logits = model(x)                                # forward pass
            loss = nn.functional.cross_entropy(logits, y)    # how wrong were we
            loss.backward()                                  # gradients, by backprop
            opt.step()                                       # nudge the weights
            # -----------------------------------------------------

            running_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)

        val = evaluate(model, val_loader, device)
        history["train_loss"].append(running_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])

        # Keep the best weights we've seen so far.
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, name + ".pt"))

        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  "
                  f"train loss {history['train_loss'][-1]:.3f} "
                  f"acc {history['train_acc'][-1]:.3f}   |   "
                  f"val loss {val['loss']:.3f} acc {val['acc']:.3f}"
                  + ("  <-- best" if val["acc"] == best_val_acc else ""))

    history["best_val_acc"] = best_val_acc
    history["minutes"] = (time.time() - start) / 60

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"  done in {history['minutes']:.1f} min. "
              f"best val acc = {best_val_acc:.3f}  (saved as '{name}')")
    return history


def load_best(model, name, device=None):
    """Load the best saved weights of a finished run back into a model."""
    device = device or get_device()
    path = os.path.join(CHECKPOINT_DIR, name + ".pt")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


# =============================================================================
# 4. RESULTS
# =============================================================================

def load_results(names=None):
    """
    Read saved runs back as a pandas table -- this is your report's results table.

    Because it reads from disk, it still works after a kernel restart, and it
    works no matter which order you ran the experiments in.
    """
    import pandas as pd

    if names is None:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    else:
        files = [os.path.join(RESULTS_DIR, n + ".json") for n in names]

    rows = []
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            h = json.load(f)
        rows.append({
            "run": h["name"],
            "best val acc": round(h["best_val_acc"], 4),
            "final train acc": round(h["train_acc"][-1], 4),
            "optimizer": h["optimizer"],
            "lr": h["lr"],
            "params": h["n_params"],
            "epochs": h["epochs"],
            "minutes": round(h["minutes"], 1),
        })
    return pd.DataFrame(rows).sort_values("best val acc", ascending=False)


def load_history(name):
    """Read one saved run's per-epoch history."""
    with open(os.path.join(RESULTS_DIR, name + ".json")) as f:
        return json.load(f)


# =============================================================================
# 5. SALIENCY
# =============================================================================

def saliency(model, images, labels=None, device=None):
    """
    Vanilla saliency map (Simonyan et al. 2013).

    Question it answers: "which pixels would change this score the most?"

    Normally backprop computes the gradient of the loss w.r.t. the *weights*.
    Here we instead compute the gradient of the class score w.r.t. the *input
    pixels*, and keep the weights fixed. A big gradient at a pixel means
    nudging that pixel would move the score a lot, i.e. the model is using it.

    We take the absolute value (we care about influence, not its direction)
    and the max over the 3 colour channels, giving one heat map per image.

    Args:
        images: normalised batch, shape (N, 3, H, W)
        labels: which class score to explain. Defaults to the predicted class.
    Returns:
        (N, H, W) tensor of saliency values
    """
    device = device or get_device()
    model.to(device).eval()

    x = images.clone().to(device)
    x.requires_grad_(True)  # tell autograd to track gradients for the INPUT

    logits = model(x)
    if labels is None:
        labels = logits.argmax(dim=1)
    labels = labels.to(device)

    # Sum the chosen class score over the batch. Because each image's score
    # only depends on its own pixels, one backward pass gives every image its
    # own correct gradient.
    score = logits.gather(1, labels.view(-1, 1)).sum()
    score.backward()

    return x.grad.abs().max(dim=1).values.cpu()


def smoothgrad(model, images, labels=None, n_samples=50, noise_level=0.15,
               device=None):
    """
    SmoothGrad (Smilkov et al. 2017): average the saliency of many noisy copies.

    Why it helps: a plain saliency map is visually noisy, because the gradient
    of a deep network fluctuates sharply from pixel to pixel. Averaging over
    nearby inputs smooths those fluctuations out and leaves the structure.

    WHY ADDING GAUSSIAN NOISE = SAMPLING FROM A NORMAL CENTRED ON THE IMAGE
    (the report asks you to explain this):
        Let the image be a fixed vector x, and let e ~ N(0, sigma^2 I) be
        Gaussian noise. Form x' = x + e. Adding a constant to a normal random
        variable shifts its mean and leaves its variance alone, so
            x' ~ N(x, sigma^2 I).
        That is exactly a sample from a normal distribution centred at the
        image, with sigma controlling how far around x we look. So SmoothGrad
        is a Monte-Carlo estimate of the average saliency in a neighbourhood
        of x, rather than the saliency at the single point x.

    sigma is a hyperparameter worth varying in the report: too small and you
    get the noisy original map back, too large and you blur away real detail.

    Args:
        noise_level: sigma as a fraction of the image's value range,
                     which is the usual way to make it scale-independent.
    """
    device = device or get_device()
    model.to(device).eval()
    images = images.to(device)

    # Scale sigma to the actual data range (our tensors are normalised, so the
    # range isn't [0,1] and a fixed sigma would mean different things per image).
    sigma = noise_level * (images.max() - images.min()).item()

    # Fix the class *before* the loop. If we let each noisy copy pick its own
    # predicted class, we'd be averaging explanations of different questions.
    if labels is None:
        with torch.no_grad():
            labels = model(images).argmax(dim=1)

    total = torch.zeros(images.shape[0], images.shape[2], images.shape[3])
    for _ in range(n_samples):
        noisy = images + torch.randn_like(images) * sigma
        total += saliency(model, noisy, labels, device)

    return total / n_samples


# =============================================================================
# 6. PLOTTING
# =============================================================================

def _save(fig, filename):
    """Save a figure into figures/ so it's ready to drop into the report."""
    if filename:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")


def plot_curves(names, title="", filename=None):
    """
    Plot training and validation curves for one or more runs side by side.

    Reading these is most of the analysis in your report:
      - train accuracy rising while val accuracy falls  -> overfitting
      - both flat and low                               -> underfitting, or lr too low
      - val curve very jagged                           -> lr too high
    """
    if isinstance(names, str):
        names = [names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        h = load_history(name)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], "--", alpha=0.6)
        axes[0].plot(epochs, h["val_loss"], label=name,
                     color=axes[0].lines[-1].get_color())
        axes[1].plot(epochs, h["train_acc"], "--", alpha=0.6)
        axes[1].plot(epochs, h["val_acc"], label=name,
                     color=axes[1].lines[-1].get_color())

    for ax, ylabel in zip(axes, ["loss", "accuracy"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("loss (dashed = train, solid = validation)")
    axes[1].set_title("accuracy (dashed = train, solid = validation)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def show_images(images, titles=None, ncols=7, filename=None, title=""):
    """Show a grid of normalised image tensors (undoes the normalisation first)."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(denormalize(images[i]).permute(1, 2, 0).numpy())
            if titles is not None:
                ax.set_title(titles[i], fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def show_mistakes(model, loader, n=14, most_confident=True, filename=None,
                  device=None):
    """
    Show the test images the model got wrong -- the report asks which images
    are misclassified and whether there's an obvious reason.

    By default it shows the mistakes the model was most *confident* about,
    which are the interesting ones: they tend to reveal what the model has
    actually latched onto (buns, red/yellow sauce, plates) rather than just
    being genuinely ambiguous pictures.
    """
    result = evaluate(model, loader, device)
    wrong = (result["preds"] != result["labels"]).nonzero().squeeze(1)

    # Sort the wrong ones by how confident the model was in its wrong answer.
    confidence = result["probs"][wrong].max(dim=1).values
    order = confidence.argsort(descending=most_confident)
    wrong = wrong[order][:n]

    # Pull the actual images back out of the underlying dataset.
    images = torch.stack([loader.dataset[i][0] for i in wrong.tolist()])
    titles = [
        f"said: {CLASS_NAMES[result['preds'][i]]}\n"
        f"true: {CLASS_NAMES[result['labels'][i]]} ({result['probs'][i].max():.2f})"
        for i in wrong.tolist()
    ]
    return show_images(images, titles, filename=filename,
                       title=f"misclassified test images "
                             f"({len(result['preds']) - int((result['preds'] == result['labels']).sum())} wrong in total)")


def show_saliency(model, images, labels=None, use_smoothgrad=True,
                  noise_level=0.15, n_samples=50, filename=None, device=None):
    """
    Show images on the top row and their saliency maps underneath.

    A good saliency map should light up the hotdog itself. If instead it lights
    up the plate, the background or the person holding it, the model is using a
    shortcut -- which is exactly the kind of finding worth reporting.
    """
    if use_smoothgrad:
        maps = smoothgrad(model, images, labels, n_samples, noise_level, device)
        kind = f"SmoothGrad (sigma={noise_level}, {n_samples} samples)"
    else:
        maps = saliency(model, images, labels, device)
        kind = "vanilla saliency"

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(denormalize(images[i]).permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        # Clip the top of the colour range: a few extreme pixels would
        # otherwise wash the whole map out.
        m = maps[i].numpy()
        axes[1, i].imshow(m, cmap="hot", vmax=np.percentile(m, 99))
        axes[1, i].axis("off")
    fig.suptitle(kind)
    fig.tight_layout()
    _save(fig, filename)
    return fig
