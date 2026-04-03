"""
Process Expert Validation Results
===================================
Hitung Cohen's Kappa dan update dataset dengan label yang dikoreksi ahli.

Usage:
    python src/tools/process_validation_results.py
    python src/tools/process_validation_results.py --data-dir data/validation_stratified_5pct
"""

import json
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def cohens_kappa(y_true, y_pred):
    """Hitung Cohen's Kappa untuk inter-rater agreement."""
    n = len(y_true)
    if n == 0:
        return 0.0

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def interpret_kappa(k):
    """Interpretasi nilai Cohen's Kappa."""
    if k < 0:
        return "Poor"
    elif k < 0.20:
        return "Slight"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Moderate"
    elif k < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def main():
    parser = argparse.ArgumentParser(description="Process Validation Results")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    # Find data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        for name in ["validation_stratified_5pct", "validation_stratified_10pct",
                      "validation_full_1938", "validation"]:
            p = Path(f"data/{name}")
            if (p / "expert_results.json").exists():
                data_dir = p
                break
        else:
            print("ERROR: Tidak ditemukan expert_results.json")
            return

    results_path = data_dir / "expert_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} tidak ditemukan. Jalankan validasi dulu.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print("=" * 60)
    print("PROCESS EXPERT VALIDATION RESULTS")
    print(f"  Data: {data_dir}")
    print(f"  Validated: {len(results)} samples")
    print("=" * 60)

    # 1. Collect auto vs expert labels
    auto_labels = []
    expert_labels = []
    corrections = []

    for key, r in results.items():
        auto = r["auto_label"]
        expert = auto if r["expert_label"] == "agree" else r["expert_label"]
        auto_labels.append(auto)
        expert_labels.append(expert)

        if auto != expert:
            corrections.append({
                "no": key,
                "auto_label": auto,
                "expert_label": expert,
                "notes": r.get("notes", ""),
            })

    # 2. Cohen's Kappa
    kappa = cohens_kappa(auto_labels, expert_labels)
    interpretation = interpret_kappa(kappa)

    print(f"\n[1] Inter-Rater Agreement")
    print(f"  Cohen's Kappa: {kappa:.4f} ({interpretation})")
    print(f"  Agreement: {len(results) - len(corrections)}/{len(results)} "
          f"({(len(results) - len(corrections)) / len(results) * 100:.1f}%)")
    print(f"  Corrections: {len(corrections)}")

    # 3. Correction details
    if corrections:
        print(f"\n[2] Koreksi Detail:")
        correction_matrix = Counter()
        for c in corrections:
            correction_matrix[(c["auto_label"], c["expert_label"])] += 1

        print(f"  {'Auto Label':>12s} -> {'Expert Label':<12s} : Count")
        for (auto, expert), count in correction_matrix.most_common():
            print(f"  {auto:>12s} -> {expert:<12s} : {count}")

    # 4. Per-class agreement
    print(f"\n[3] Agreement per Emosi:")
    for emo in EMOTIONS:
        emo_auto = [a for a, e in zip(auto_labels, expert_labels) if a == emo]
        emo_expert = [e for a, e in zip(auto_labels, expert_labels) if a == emo]
        if emo_auto:
            agree = sum(1 for a, e in zip(emo_auto, emo_expert) if a == e)
            print(f"  {emo:>10s}: {agree}/{len(emo_auto)} ({agree/len(emo_auto)*100:.0f}%)")

    # 5. Save report
    report = {
        "total_validated": len(results),
        "total_corrections": len(corrections),
        "agreement_rate": (len(results) - len(corrections)) / len(results),
        "cohens_kappa": round(kappa, 4),
        "kappa_interpretation": interpretation,
        "corrections": corrections,
    }
    report_path = data_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
