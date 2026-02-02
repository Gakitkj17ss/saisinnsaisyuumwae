#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phoneme-level analysis (designed for consonants)
- Input: prediction/target lists (both are "phoneme sequences")
- Dynamic programming alignment (visualizes insertion/deletion/substitution)
- Per-consonant accuracy, support count, confusion matrix, and representative confusion pairs
- Save as CSV/PNG (optional)

Usage:
    from phoneme_analysis_unified import analyze_phonemes_unified
    result = analyze_phonemes_unified(predictions, targets, labels=encoder.consonants, save_dir="results/phoneme_analysis")
"""

from __future__ import annotations
import os
import math
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Alignment (Levenshtein DP)
# =========================================================
def _align_sequences(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align ref (target) and hyp (prediction) with minimum edit distance
    Returns: [(r, h), ...]  (r==None: insertion, h==None: deletion, otherwise: match/substitution)
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    bt = np.zeros((n+1, m+1), dtype=np.int8)  # 0:diag, 1:up(del), 2:left(ins)

    for i in range(1, n+1): dp[i,0] = i; bt[i,0] = 1
    for j in range(1, m+1): dp[0,j] = j; bt[0,j] = 2

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_sub = 0 if ref[i-1] == hyp[j-1] else 1
            a = dp[i-1, j-1] + cost_sub      # diag (match/sub)
            b = dp[i-1, j] + 1               # up (del)
            c = dp[i, j-1] + 1               # left (ins)
            best = a; code = 0
            if b < best: best, code = b, 1
            if c < best: best, code = c, 2
            dp[i,j] = best; bt[i,j] = code

    # backtrack
    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        code = bt[i,j]
        if i>0 and j>0 and code == 0:
            aligned.append((ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i>0 and (j==0 or code == 1):
            aligned.append((ref[i-1], None)); i -= 1
        else:
            aligned.append((None, hyp[j-1])); j -= 1
    return aligned[::-1]


# =========================================================
# Main analysis
# =========================================================
def analyze_phonemes_unified(
    predictions: List[List[str]],
    targets: List[List[str]],
    labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    top_k: int = 5,
    plot_confusion: bool = True,
    apply_collapse: bool = False,   # For backward compatibility (ignored)
    **_ignored,                     # Ignore unexpected extra arguments
) -> Dict:

    """
    Arguments:
        predictions: List of predicted phoneme sequences (e.g., [['k','t'], ['s','n',...], ...])
        targets    : List of target phoneme sequences
        labels     : Order of phoneme labels (if not specified, created from union of targets/preds)
        save_dir   : Directory to save CSV/PNG (None: don't save)
        top_k      : Extract top k confusion pairs
        plot_confusion: Whether to output confusion matrix PNG

    Returns (main keys):
        {
          'overall_accuracy': float (0-1),
          'macro_accuracy': float (0-1),
          'per_phoneme': { 'k': {'support':..,'correct':..,'accuracy':..}, ... },
          'confusion': 2D list (pred×true count matrix),
          'labels': [..],
          'top_confusions': { 'k': [('t', 10), ('g', 7), ...], ... }
        }
    """
    assert len(predictions) == len(targets), "predictions/targets count mismatch"

    # Label set
    if labels is None:
        label_set = set()
        for seq in predictions: label_set.update(seq)
        for seq in targets: label_set.update(seq)
        labels = sorted([x for x in label_set if x is not None])
    L = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    # Accumulators
    support = np.zeros(L, dtype=np.int64)  # Target occurrence count (denominator)
    correct = np.zeros(L, dtype=np.int64)  # Match count
    conf = np.zeros((L, L), dtype=np.int64)  # [pred, true] count
    # confusions_by_true[true_lab][pred_lab] = count
    confusions_by_true: Dict[str, Dict[str, int]] = {lab: {} for lab in labels}

    # Align and accumulate for each sample
    for tgt_seq, pred_seq in zip(targets, predictions):
        aligned = _align_sequences(tgt_seq, pred_seq)
        for r, h in aligned:
            if r is not None:
                # Appeared in target
                if r in idx: support[idx[r]] += 1
            if (r is not None) and (h is not None):
                # Substitution/match
                if (r in idx) and (h in idx):
                    conf[idx[h], idx[r]] += 1
                    if r == h:
                        correct[idx[r]] += 1
                    else:
                        confusions_by_true[r][h] = confusions_by_true[r].get(h, 0) + 1
            elif (r is not None) and (h is None):
                # deletion: no prediction -> not counted in confusion matrix, only support increases
                pass
            elif (r is None) and (h is not None):
                # insertion: prediction without target -> skip (no "other" row in confusion matrix)
                pass

    # Metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        per_acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=np.float64), where=support>0)
    overall_acc = (correct.sum() / support.sum()) if support.sum() > 0 else 0.0
    macro_acc = per_acc[support>0].mean() if (support>0).any() else 0.0

    # Top confusions
    top_confusions = {}
    for r in labels:
        d = confusions_by_true[r]
        ranked = sorted(d.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        top_confusions[r] = ranked

    result = {
        "overall_accuracy": float(overall_acc),
        "macro_accuracy": float(macro_acc),
        "per_phoneme": {
            lab: {
                "support": int(support[i]),
                "correct": int(correct[i]),
                "accuracy": float(per_acc[i]) if support[i] > 0 else None,
            }
            for i, lab in enumerate(labels)
        },
        "confusion": conf.astype(int).tolist(),  # pred × true
        "labels": labels,
        "top_confusions": {k: [(p, int(c)) for (p, c) in v] for k, v in top_confusions.items()},
    }

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # per-phoneme CSV
        _save_per_phoneme_csv(result, os.path.join(save_dir, "per_phoneme_metrics.csv"))
        # confusion CSV
        _save_confusion_csv(conf, labels, os.path.join(save_dir, "confusion_matrix.csv"))
        # overview JSON
        with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        # plot
        if plot_confusion:
            _plot_confusion(conf, labels, os.path.join(save_dir, "confusion_matrix.png"))

    return result


# =========================================================
# Save utilities
# =========================================================
def _save_per_phoneme_csv(result: Dict, path: str):
    labels = result["labels"]
    rows = [["phoneme", "support", "correct", "accuracy"]]
    for lab in labels:
        d = result["per_phoneme"][lab]
        rows.append([lab, d["support"], d["correct"], None if d["accuracy"] is None else round(d["accuracy"], 4)])
    _write_csv(rows, path)


def _save_confusion_csv(conf: np.ndarray, labels: List[str], path: str):
    # Header row: ['', true1, true2, ...]
    header = ["pred\\true"] + labels
    rows = [header]
    for i, pred_lab in enumerate(labels):
        row = [pred_lab] + list(map(int, conf[i].tolist()))
        rows.append(row)
    _write_csv(rows, path)


def _write_csv(rows: List[List], path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _plot_confusion(conf: np.ndarray, labels: List[str], path: str):
    """Simple heatmap with matplotlib only"""
    if conf.sum() == 0:
        # Avoid empty plot
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        plt.axis("off")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    # Column-normalized ratio shows confusion trends more clearly
    col_sum = conf.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = conf / np.where(col_sum == 0, 1, col_sum)

    fig, ax = plt.subplots(figsize=(1.2*len(labels), 1.0*len(labels)))
    im = ax.imshow(norm, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(conf[i, j])
            if count > 0:
                text_color = "white" if norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(count), ha="center", va="center", 
                       color=text_color, fontsize=9, weight="bold")
    
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (column-normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)