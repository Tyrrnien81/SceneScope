# Vision Detection Algorithm — Inspection-Oriented Analysis

This document reframes a scene-classification model as a **vision inspection /
detection system** and analyzes it with the metrics used to tune real
machine-vision inspection equipment: detection power (precision/recall),
failure analysis (confusion), and the false-reject vs. missed-detection
operating-point trade-off.

> **Scope (honest framing):** The dataset is MiniPlaces (100 scene categories,
> 32×32 images) — an academic benchmark, not battery-line imagery. What
> transfers directly is the *methodology*: building/optimizing a detection
> algorithm, quantifying detection power, doing data-driven failure analysis,
> and tuning the operating point to balance Overkill vs. NG Leakage.

## 1. Algorithm development & optimization

| Model | top-1 | top-5 | macro P / R / F1 | params |
|-------|:---:|:---:|:---:|:---:|
| LeNet-5 baseline | 19.4% | — | — | 0.15M |
| **ImprovedNet (custom ResNet)** | **49.0%** | **77.4%** | **49.3 / 49.0 / 48.9%** | 2.8M |

A **2.5× improvement in detection accuracy** was achieved by developing a new
algorithm (residual blocks + batch normalization), then optimizing it with data
augmentation, an SGD warmup→cosine learning-rate schedule, label smoothing, and
weight decay. A baseline failure mode (training divergence at high learning
rate) was diagnosed and resolved through scheduled, normalized training.

## 2. Detection power (per-class precision / recall)

The model is well balanced (macro precision ≈ recall), i.e. no class collapse.

- **Strongest detection** (visually distinctive scenes): `track` F1 84.8%,
  `cockpit` 82.7%, `bamboo_forest` 82.0%, `boxing_ring` 80.2%, `swimming_pool` 77.8%.
- **Weakest detection** (ambiguous scenes): `coffee_shop` F1 17.5%, `museum`
  17.7%, `river` 18.1%, `restaurant` 20.7%, `valley` 22.2%.

## 3. Failure analysis (most-confused pairs)

The dominant errors are between **semantically similar classes** — the direct
analog of confusable defect categories on an inspection line:

| True → Predicted | Count | Root cause |
|---|:---:|---|
| shower → bathroom | 27 | overlapping visual context |
| classroom → kindergarden_classroom | 22 | near-identical layout |
| mountain → valley | 21 | shared terrain features |
| monastery → temple / abbey → church | 17 each | similar architecture |
| candy_store → supermarket | 15 | similar shelving/retail layout |

**Root cause:** at 32×32 the fine details that separate these scenes are lost.
The fix direction is higher input resolution / targeted data, not more training.

## 4. Operating-point trade-off — Overkill vs. NG Leakage

Treating low-confidence predictions as "flag for manual review" (as an inspection
machine routes uncertain units), sweeping the confidence threshold reproduces the
sensitivity trade-off engineers tune in production:

| Confidence thr | Coverage (auto) | Accuracy (auto) | **NG Leakage** (auto error) | **Overkill** (flagged) |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 100.0% | 49.0% | 51.0% | 0.0% |
| 0.30 | 71.1% | 60.7% | 39.3% | 28.9% |
| 0.50 | 48.5% | 72.8% | 27.2% | 51.5% |
| 0.70 | 32.6% | 82.6% | 17.4% | 67.4% |
| 0.90 | 19.0% | 91.5% | **8.5%** | 81.0% |
| 0.95 | 13.6% | 94.4% | **5.6%** | 86.4% |

**Reading:** raising the threshold cuts leakage (8.5% at 0.90) but raises overkill
(81% flagged) — the same yield-vs-quality balance set on real inspection equipment.
The right operating point depends on the relative cost of an escaped defect vs. a
falsely-rejected good unit. See `outputs/risk_coverage.png` and
`outputs/confusion_matrix.png`.

## 5. Engineering

- **Stack:** Python, PyTorch (`nn.Module`), torchvision.
- **Compute:** trained end-to-end on Apple Silicon GPU (PyTorch MPS), ~66 min /
  50 epochs over 100K images.
- **Reproducibility & isolation:** isolated `uv`/`venv` environment; baseline
  graded code left untouched; all artifacts additive.
- **Reproduce:** `python train_improved.py --epochs 50` then
  `python inspection_eval.py`.
