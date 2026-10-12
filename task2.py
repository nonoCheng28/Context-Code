import argparse
import csv
from collections import Counter, defaultdict
from math import fsum, log2
from pathlib import Path


def estimate_conditional_entropy(text, k):
    """用位置 i=k,...,N-1 估计 H(X_i | X_{i-k},...,X_{i-1})。

    每阶使用 N-k 个样本，不填充开头、不循环连接结尾。
    k=0 时 context 是空字符串，结果就是 Task 1 的零阶熵。
    """
    if not 0 <= k < len(text):
        raise ValueError("k 必须满足 0 <= k < 文本字符数")

    # counts[context][char] = 上下文 context 后出现 char 的次数。
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        context = text[i - k:i]
        char = text[i]
        counts[context][char] += 1

    sample_count = len(text) - k
    weighted_entropies = []
    singleton_contexts = 0
    deterministic_samples = 0

    for next_counts in counts.values():
        context_count = sum(next_counts.values())

        # 固定一个上下文 c，计算 H(X | c)。
        context_entropy = -fsum(
            (count / context_count) * log2(count / context_count)
            for count in next_counts.values()
        )

        # P(c) = context_count / sample_count，对各上下文的熵加权。
        weighted_entropies.append(
            (context_count / sample_count) * context_entropy
        )

        if context_count == 1:
            singleton_contexts += 1
        if len(next_counts) == 1:
            # 只观察到一种后继字符的上下文，其经验条件熵为 0。
            deterministic_samples += context_count

    context_count = len(counts)
    return {
        "k": k,
        "entropy_bits_per_character": fsum(weighted_entropies),
        "sample_count": sample_count,
        "context_count": context_count,
        "singleton_context_count": singleton_contexts,
        "singleton_context_fraction": singleton_contexts / context_count,
        "singleton_sample_fraction": singleton_contexts / sample_count,
        "deterministic_sample_fraction": deterministic_samples / sample_count,
    }


def plot_entropy(results, output_dir):
    import matplotlib

    matplotlib.use("Agg")  # 保存图像，无需打开交互窗口。
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    orders = [row["k"] for row in results]
    entropies = [row["entropy_bits_per_character"] for row in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    ax.plot(orders, entropies, marker="o", color="#0072B2", linewidth=1.8)
    ax.set_xlabel("Context length k (characters)")
    ax.set_ylabel("Empirical conditional entropy (bits/character)")
    ax.set_title("Conditional entropy of Game of Thrones")
    ax.set_xticks(orders)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(output_dir / "conditional_entropy.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=5, help="最大上下文长度，默认 5")
    args = parser.parse_args()

    # 相对于脚本定位文件，在项目目录外运行也能找到原文。
    base_dir = Path(__file__).resolve().parent
    text = (base_dir / "GameOfThrones.txt").read_bytes().decode("gb18030", errors="strict")
    if not 0 <= args.max_k < len(text):
        parser.error("--max-k 必须满足 0 <= max_k < 文本字符数")

    output_dir = base_dir / "task2_results"
    output_dir.mkdir(exist_ok=True)
    print(f"字符数：{len(text)}；字符种类：{len(set(text))}", flush=True)
    print("保留大小写、标点、空格及原始 CR/LF 换行字符。", flush=True)

    results = []
    for k in range(args.max_k + 1):
        row = estimate_conditional_entropy(text, k)
        results.append(row)
        print(
            f"k={k}: H_k={row['entropy_bits_per_character']:.9f} bit/字符，"
            f"上下文数={row['context_count']}，"
            f"单次上下文占比={row['singleton_context_fraction']:.2%}",
            flush=True,
        )

    csv_path = output_dir / "conditional_entropy.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    nonincreasing = all(
        current["entropy_bits_per_character"] <= previous["entropy_bits_per_character"] + 1e-12
        for previous, current in zip(results, results[1:])
    )
    print(f"当前数据中 H_k 是否单调不增：{nonincreasing}")
    plot_entropy(results, output_dir)
    print(f"统计表和曲线已保存至：{output_dir}")


if __name__ == "__main__":
    main()
