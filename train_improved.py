"""
Train ImprovedNet (CIFAR-style ResNet) on MiniPlaces with MPS/GPU acceleration.

Additive to the assignment: does NOT touch student_code.py or results.txt.
Reuses the provided MiniPlaces dataloader and reports top-1 / top-5 on the
labeled validation set (10K images). The MiniPlaces test set has no public
labels, so it cannot be scored locally.

Examples:
    .venv/bin/python train_improved.py --epochs 50
    .venv/bin/python train_improved.py --eval-only --resume outputs/improved_best.pth.tar
"""

import os
# Let any op without an MPS kernel transparently fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataloader import MiniPlaces
from improved_model import ImprovedNet, count_model_params

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CKPT_DIR = "./outputs"
BEST_PATH = os.path.join(CKPT_DIR, "improved_best.pth.tar")
HISTORY_PATH = os.path.join(CKPT_DIR, "improved_history.json")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(batch_size, num_workers):
    # Augmentation only on the training split; val uses a clean transform.
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    train_set = MiniPlaces(root="./data", split="train", download=False, transform=train_tf)
    val_set = MiniPlaces(root="./data", split="val", download=False, transform=val_tf)

    persistent = num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=persistent)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (top1 %, top5 %) on the given loader."""
    model.eval()
    top1, top5, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred5 = out.topk(5, dim=1, largest=True, sorted=True)   # [N, 5]
        pred5 = pred5.t()                                          # [5, N]
        correct = pred5.eq(y.view(1, -1).expand_as(pred5))         # [5, N] bool
        top1 += correct[0].sum().item()
        top5 += correct[:5].reshape(-1).sum().item()
        total += y.size(0)
    return 100.0 * top1 / total, 100.0 * top5 / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, total = 0.0, 0
    bar = tqdm(loader, total=len(loader), disable=not sys.stdout.isatty())
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        total += y.size(0)
    return running_loss / total


def main():
    parser = argparse.ArgumentParser(description="Train ImprovedNet on MiniPlaces")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup", default=5, type=int, help="linear warmup epochs")
    parser.add_argument("--lr", default=0.1, type=float, help="peak learning rate")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = get_device()

    model = ImprovedNet().to(device)
    n_params = count_model_params(model)
    print(f"Device: {device} | ImprovedNet params: {n_params:,} ({n_params/1e6:.3f}M)")

    train_loader, val_loader = build_loaders(args.batch_size, args.num_workers)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ----- eval-only path -----
    if args.eval_only:
        if not args.resume or not os.path.isfile(args.resume):
            print(f"--eval-only requires a valid --resume checkpoint (got '{args.resume}')")
            return
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        t0 = time.time()
        top1, top5 = evaluate(model, val_loader, device)
        print(f"[Eval] checkpoint epoch {ckpt.get('epoch','?')} | "
              f"top-1: {top1:.2f}% | top-5: {top5:.2f}% | {time.time()-t0:.1f}s")
        return

    # ----- training path -----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup])

    best_top1, history = 0.0, []
    print(f"Training for {args.epochs} epochs (warmup {args.warmup}, peak lr {args.lr})\n")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        top1, top5 = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        is_best = top1 > best_top1
        best_top1 = max(best_top1, top1)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {train_loss:.3f} | "
              f"val top-1 {top1:5.2f}% | top-5 {top5:5.2f}% | "
              f"lr {lr_now:.4f} | {dt:5.1f}s{'  <-- best' if is_best else ''}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_top1": top1, "val_top5": top5,
                        "lr": lr_now, "sec": dt})
        if is_best:
            torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                        "best_top1": best_top1}, BEST_PATH)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nFinished. Best val top-1: {best_top1:.2f}%  (saved to {BEST_PATH})")


if __name__ == "__main__":
    main()
