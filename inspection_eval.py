"""
Inspection-oriented evaluation of the trained ImprovedNet.

Reframes the scene classifier as a vision *detection* system and reports the
metrics a manufacturing machine-vision engineer cares about:

  * Per-class precision / recall / F1 and macro averages -> "detection power"
  * Confusion matrix + most-confused class pairs           -> failure analysis
  * Confidence-threshold (selective classification) sweep  -> the false-reject
    (Overkill) vs. missed-detection (NG Leakage) trade-off used to tune an
    inspection machine's operating point.

No additional training. Uses only numpy/torch (+ optional matplotlib for plots),
loading outputs/improved_best.pth.tar and the labeled val split (10K images).
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import numpy as np
import torch
import torchvision.transforms as T

from dataloader import MiniPlaces
from improved_model import ImprovedNet

NUM_CLASSES = 100
CKPT = "outputs/improved_best.pth.tar"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_class_names(path="data/miniplaces/train.txt"):
    """Map label id -> human category name from train paths (train/<l>/<name>/..)."""
    id_to_name = {}
    with open(path) as f:
        for line in f:
            rel, lid = line.split()
            lid = int(lid)
            if lid not in id_to_name:
                id_to_name[lid] = rel.split("/")[2]   # the category folder
    return [id_to_name.get(i, str(i)) for i in range(NUM_CLASSES)]


@torch.no_grad()
def collect_predictions(model, loader, device):
    probs_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), dim=1)
        probs_all.append(p.cpu())
        labels_all.append(y)
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return probs, labels


def per_class_metrics(cm):
    """Precision/recall/F1 per class from a confusion matrix (rows=true, cols=pred)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)       # actual occurrences per class
    predicted = cm.sum(axis=0).astype(float)     # times predicted per class
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def main():
    device = get_device()
    names = load_class_names()

    val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    val = MiniPlaces(root="./data", split="val", download=False, transform=val_tf)
    loader = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False)

    model = ImprovedNet().to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs, labels = collect_predictions(model, loader, device)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    # ---- confusion matrix + detection power ----
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    precision, recall, f1, support = per_class_metrics(cm)
    top1 = np.trace(cm) / cm.sum()

    print(f"Device: {device} | checkpoint epoch {ckpt.get('epoch','?')}")
    print(f"Overall top-1 accuracy : {100*top1:.2f}%")
    print(f"Macro precision        : {100*precision.mean():.2f}%")
    print(f"Macro recall           : {100*recall.mean():.2f}%")
    print(f"Macro F1               : {100*f1.mean():.2f}%")

    order = np.argsort(f1)
    print("\nWeakest 5 classes (lowest F1 = hardest to detect):")
    for i in order[:5]:
        print(f"  {names[i]:<18} P {100*precision[i]:5.1f}%  R {100*recall[i]:5.1f}%  F1 {100*f1[i]:5.1f}%")
    print("Strongest 5 classes (highest F1):")
    for i in order[::-1][:5]:
        print(f"  {names[i]:<18} P {100*precision[i]:5.1f}%  R {100*recall[i]:5.1f}%  F1 {100*f1[i]:5.1f}%")

    # ---- most-confused class pairs (off-diagonal) ----
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat = np.argsort(cm_off.ravel())[::-1][:8]
    print("\nTop confused pairs (true -> predicted):")
    confused = []
    for idx in flat:
        t, p = divmod(int(idx), NUM_CLASSES)
        print(f"  {names[t]:<18} -> {names[p]:<18} : {cm_off[t, p]} times")
        confused.append({"true": names[t], "pred": names[p], "count": int(cm_off[t, p])})

    # ---- selective classification: Overkill vs NG Leakage trade-off ----
    print("\nConfidence-threshold sweep (inspection operating points):")
    print("  thr  | coverage(auto) | accuracy(auto) | leakage(auto err) | flagged(overkill)")
    sweep = []
    for thr in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]:
        mask = conf >= thr
        coverage = mask.mean()
        if mask.any():
            acc = (preds[mask] == labels[mask]).mean()
        else:
            acc = float("nan")
        leakage = 1 - acc
        flagged = 1 - coverage
        print(f"  {thr:0.2f} |    {100*coverage:5.1f}%     |    {100*acc:5.1f}%     |"
              f"      {100*leakage:5.1f}%      |     {100*flagged:5.1f}%")
        sweep.append({"threshold": thr, "coverage": coverage,
                      "accuracy": acc, "leakage": leakage, "flagged": flagged})

    # ---- persist metrics ----
    os.makedirs("outputs", exist_ok=True)
    metrics = {
        "top1": top1, "macro_precision": precision.mean(),
        "macro_recall": recall.mean(), "macro_f1": f1.mean(),
        "weakest_classes": [names[i] for i in order[:5]],
        "strongest_classes": [names[i] for i in order[::-1][:5]],
        "confused_pairs": confused, "threshold_sweep": sweep,
    }
    with open("outputs/inspection_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved -> outputs/inspection_metrics.json")

    # ---- optional plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrix heatmap (log-scaled for readability).
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(np.log1p(cm), cmap="viridis")
        ax.set_title("Confusion matrix (log scale)")
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        fig.colorbar(im, ax=ax, label="log(1 + count)")
        fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png", dpi=130)

        # Risk-coverage curve: accuracy on auto-decided items vs coverage.
        thr_grid = np.linspace(0, 0.99, 100)
        covs, accs = [], []
        for thr in thr_grid:
            m = conf >= thr
            if m.sum() < 10:
                break
            covs.append(m.mean())
            accs.append((preds[m] == labels[m]).mean())
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(np.array(covs) * 100, np.array(accs) * 100, marker=".")
        ax2.set_xlabel("Coverage: auto-decided units (%)")
        ax2.set_ylabel("Accuracy on auto-decided units (%)")
        ax2.set_title("Risk-coverage: Overkill vs. NG-Leakage trade-off")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig("outputs/risk_coverage.png", dpi=130)
        print("Saved -> outputs/confusion_matrix.png, outputs/risk_coverage.png")
    except ImportError:
        print("(matplotlib not installed; skipped plots)")


if __name__ == "__main__":
    main()
