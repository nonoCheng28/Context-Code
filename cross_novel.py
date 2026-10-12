"""Cross-novel extension: theoretical cross-entropy, not actual Huffman files.

Run: python cross_novel.py
Optional: python cross_novel.py --orders 2 3 --alpha 0.1
Only Python standard-library dependencies. Rates exclude the first k characters
and model storage. Both self and cross evaluations use the same smoothing rule.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import fsum, isclose, isfinite, log2
from pathlib import Path


# 1. Counts use exactly the same N-k positions as Task2.
def count_contexts(text, k):
    if not 1 <= k < len(text):
        raise ValueError("Require 1 <= k < text length")
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        counts[text[i-k:i]][text[i]] += 1
    return dict(counts)


# 2. Evaluate a trained model on a separate empirical test distribution.
def evaluate(train, test, alphabet_size, alpha):
    if not isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    sample_count = sum(sum(row.values()) for row in test.values())
    entropy_terms, cross_terms, kl_terms = [], [], []
    unseen_context_samples = 0
    unseen_successor_samples = 0
    for context, observed in test.items():
        test_total = sum(observed.values())
        trained = train.get(context, {})
        train_total = sum(trained.values())
        # For an unseen context this gives a uniform distribution, 1/|Sigma|.
        denominator = train_total + alpha * alphabet_size
        if context not in train:
            unseen_context_samples += test_total
        for char, count in observed.items():
            p = count / test_total
            q = (trained.get(char, 0) + alpha) / denominator
            weight = count / sample_count
            entropy_terms.append(-weight * log2(p))
            cross_terms.append(-weight * log2(q))
            kl_terms.append(weight * log2(p / q))
            # Separate from unseen contexts, so the two diagnostics do not overlap.
            if context in train and char not in trained:
                unseen_successor_samples += count

    entropy = fsum(entropy_terms)
    cross_entropy = fsum(cross_terms)
    kl = fsum(kl_terms)
    if not isclose(cross_entropy - entropy, kl, abs_tol=1e-10):
        raise ArithmeticError("Cross-entropy minus empirical entropy must equal conditional KL")
    if kl < -1e-10:
        raise ArithmeticError("KL must be nonnegative")
    return {
        "sample_count": sample_count,
        "empirical_entropy_bpc": entropy,
        "cross_entropy_bpc": cross_entropy,
        "conditional_kl_bpc": kl,
        "unseen_context_sample_fraction": unseen_context_samples / sample_count,
        "unseen_successor_sample_fraction": unseen_successor_samples / sample_count,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    if not isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    orders = sorted(set(args.orders))
    if len(orders) < 2:
        parser.error("Use at least two orders to include the higher-order experiment")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "cross_novel_data"
    provenance = json.loads((data_dir / "provenance.json").read_text("utf-8"))
    texts, hashes = {}, {}
    for name in ("A", "B"):
        record = provenance["outputs"][name]
        data = (data_dir / record["file"]).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        if hashes[name] != record["sha256"]:
            raise ValueError(f"Input {name} differs from the recorded preprocessing output")
        texts[name] = data.decode("utf-8", errors="strict")
    if set(texts["A"]) != set(texts["B"]):
        raise ValueError("The two preprocessed texts must have identical alphabets")
    if min(orders) < 1 or max(orders) >= min(map(len, texts.values())):
        parser.error("Orders must satisfy 1 <= k < both text lengths")
    alphabet_size = len(set(texts["A"]))

    results, comparisons = [], []
    for k in orders:
        counts = {name: count_contexts(text, k) for name, text in texts.items()}
        scores = {}
        # Matches the four-row order specified in the guidance.
        for train_name, test_name in [("B", "B"), ("A", "B"), ("A", "A"), ("B", "A")]:
            score = evaluate(counts[train_name], counts[test_name], alphabet_size, args.alpha)
            scores[train_name, test_name] = score
            results.append({
                "k": k, "train": train_name, "test": test_name,
                "alpha": args.alpha, "alphabet_size": alphabet_size,
                "train_characters": len(texts[train_name]),
                "test_characters": len(texts[test_name]), **score,
            })
            print(f"k={k} R_{train_name}{test_name}={score['cross_entropy_bpc']:.9f} bits/character")

        for train_name, test_name in [("A", "B"), ("B", "A")]:
            cross = scores[train_name, test_name]
            self_score = scores[test_name, test_name]
            excess = cross["cross_entropy_bpc"] - self_score["cross_entropy_bpc"]
            # With smoothed self models, excess = cross KL - self KL, not cross KL alone.
            if not isclose(excess, cross["conditional_kl_bpc"] - self_score["conditional_kl_bpc"], abs_tol=1e-10):
                raise ArithmeticError("Smoothed excess-cost identity failed")
            previous = next((row for row in reversed(comparisons)
                             if row["direction"] == f"{train_name}->{test_name}"), None)
            comparisons.append({
                "k": k, "direction": f"{train_name}->{test_name}",
                "cross_entropy_bpc": cross["cross_entropy_bpc"],
                "self_cross_entropy_bpc": self_score["cross_entropy_bpc"],
                "excess_cost_bpc": excess,
                "test_empirical_entropy_bpc": cross["empirical_entropy_bpc"],
                "cross_conditional_kl_bpc": cross["conditional_kl_bpc"],
                "self_smoothing_penalty_bpc": self_score["conditional_kl_bpc"],
                "previous_k": previous["k"] if previous else "",
                "excess_change_from_previous_k_bpc": excess - previous["excess_cost_bpc"] if previous else "",
            })
            print(f"  {train_name}->{test_name}: excess={excess:.9f}, KL={cross['conditional_kl_bpc']:.9f}")

    output_dir = base_dir / "cross_novel_results"
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "cross_entropy.csv", results)
    write_csv(output_dir / "comparisons.csv", comparisons)
    settings = {
        "metric": "theoretical conditional cross-entropy, bits per evaluated character",
        "orders": orders, "alpha": args.alpha, "alphabet_size": alphabet_size,
        "A": "Game of Thrones", "B": "The Great Gatsby",
        "input_sha256": hashes,
        "source_commit_B": provenance["commit"],
        "evaluated_positions": "i=k,...,N-1; denominator N-k",
        "smoothing": "(n_train(context,char)+alpha)/(n_train(context)+alpha*alphabet_size)",
        "unseen_context": "uniform distribution over the shared alphabet",
        "cost_exclusions": ["first k characters", "codebook/model storage", "headers", "bit packing"],
        "self_evaluation": "in-sample self model with the same smoothing; not held-out generalization",
        "identities_verified": ["cross_entropy - empirical_entropy = conditional_KL", "cross_entropy - self_cross_entropy = cross_KL - self_KL"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
