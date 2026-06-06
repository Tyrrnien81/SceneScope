"""Generate a training-curve figure from outputs/improved_history.json.

Left panel  : validation top-1 / top-5 vs. epoch, with the LeNet baseline line.
Right panel : training loss and the cosine learning-rate schedule (twin axes),
              which explains the late-epoch accuracy gains.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LENET_BASELINE = 19.37  # best LeNet config (epochs=20) from results.txt

with open("outputs/improved_history.json") as f:
    hist = json.load(f)

epochs = [r["epoch"] for r in hist]
top1 = [r["val_top1"] for r in hist]
top5 = [r["val_top5"] for r in hist]
loss = [r["train_loss"] for r in hist]
lr = [r["lr"] for r in hist]
best = max(hist, key=lambda r: r["val_top1"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# --- Left: validation accuracy ---
ax1.plot(epochs, top1, marker=".", label="val top-1")
ax1.plot(epochs, top5, marker=".", label="val top-5")
ax1.axhline(LENET_BASELINE, ls="--", color="gray", lw=1,
            label=f"LeNet baseline ({LENET_BASELINE:.1f}%)")
ax1.annotate(f"best {best['val_top1']:.1f}%",
             xy=(best["epoch"], best["val_top1"]),
             xytext=(best["epoch"] - 16, best["val_top1"] + 4),
             arrowprops=dict(arrowstyle="->"))
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Validation accuracy")
ax1.legend(loc="lower right"); ax1.grid(True, alpha=0.3)

# --- Right: train loss + learning rate ---
ax2.plot(epochs, loss, color="tab:red", marker=".", label="train loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Train loss", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_title("Training loss & learning rate")
ax2.grid(True, alpha=0.3)
ax3 = ax2.twinx()
ax3.plot(epochs, lr, color="tab:blue", label="learning rate")
ax3.set_ylabel("Learning rate", color="tab:blue")
ax3.tick_params(axis="y", labelcolor="tab:blue")

fig.suptitle("ImprovedNet training on MiniPlaces (50 epochs, Apple MPS)")
fig.tight_layout()
fig.savefig("outputs/training_curve.png", dpi=130)
print("Saved -> outputs/training_curve.png")
