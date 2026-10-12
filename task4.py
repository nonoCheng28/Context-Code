
import csv
import json
from pathlib import Path


# 1. 读取 CSV，并用阶数 k 作为索引，避免依赖 CSV 的行顺序。
def read_rows_by_order(path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            k = int(row["k"])
            if k in rows:
                raise ValueError(f"{path.name} 中出现重复阶数 k={k}")
            rows[k] = row
    if not rows:
        raise ValueError(f"{path.name} 没有数据")
    return rows


# 2. 读取 Task1 的零阶基线；整个 metadata 文件只计费一次。
def load_task1(base_dir):
    directory = base_dir / "task1_results"
    metadata_path = directory / "huffman_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "character_count": int(metadata["character_count"]),
        "encoded_content_bytes": (directory / "GameOfThrones.huff").stat().st_size,
        "model_bytes": metadata_path.stat().st_size,
    }


# 3. 将 Task2 的 H_k 与 Task1、Task3 的文件大小按 k 合并。
def collect_results(base_dir):
    entropy_rows = read_rows_by_order(
        base_dir / "task2_results" / "conditional_entropy.csv"
    )
    compression_rows = read_rows_by_order(
        base_dir / "task3_results" / "conditional_huffman.csv"
    )
    if any(k < 1 for k in compression_rows):
        raise ValueError("Task3 应只包含正阶数，k=0 使用 Task1")
    if set(entropy_rows) != {0, *compression_rows}:
        raise ValueError("Task2 与 Task3 的阶数不一致，请先用相同最大阶数运行实验")

    baseline = load_task1(base_dir)
    character_count = baseline["character_count"]
    if character_count <= 0:
        raise ValueError("原文字符数必须大于零")

    results = []
    for k in sorted(entropy_rows):
        entropy = entropy_rows[k]
        if int(entropy["sample_count"]) != character_count - k:
            raise ValueError(f"k={k} 的 Task2 样本数与 Task1 字符数不一致")

        if k == 0:
            content_bytes = baseline["encoded_content_bytes"]
            model_bytes = baseline["model_bytes"]
        else:
            row = compression_rows[k]
            if int(row["character_count"]) != character_count:
                raise ValueError(f"k={k} 的 Task3 字符数与 Task1 不一致")
            if row["lossless_verified"].strip().lower() != "true":
                raise ValueError(f"k={k} 未通过 Task3 的无损验证")

            directory = base_dir / "task3_results"
            sizes = {
                "encoded_content_bytes": (directory / f"k{k}.huff").stat().st_size,
                "codebook_bytes": (directory / f"k{k}_codebooks.json").stat().st_size,
                "metadata_bytes": (directory / f"k{k}_metadata.json").stat().st_size,
            }
            for field, size in sizes.items():
                if size != int(row[field]):
                    raise ValueError(f"k={k} 的 {field} 与实际文件不符，请重新运行 Task3")
            if sum(sizes.values()) != int(row["total_bytes"]):
                raise ValueError(f"k={k} 的 Task3 总大小统计不一致")
            content_bytes = sizes["encoded_content_bytes"]
            model_bytes = sizes["codebook_bytes"] + sizes["metadata_bytes"]

        total_bytes = content_bytes + model_bytes
        results.append({
            "k": k,
            "entropy_bits_per_character": float(entropy["entropy_bits_per_character"]),
            "character_count": character_count,
            "encoded_content_bytes": content_bytes,
            "model_bytes": model_bytes,
            "total_bytes": total_bytes,
            "content_bits_per_character": content_bytes * 8 / character_count,
            "total_bits_per_character": total_bytes * 8 / character_count,
        })
    return results


# 4. 单坐标系比较三条曲线，保留全部阶数，不使用双轴或截断纵轴。
def plot_results(results, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    orders = [row["k"] for row in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    series = [
        ("entropy_bits_per_character", r"Empirical entropy $H_k$", "#555555", "o", "-"),
        ("content_bits_per_character", "Encoded content only", "#0072B2", "s", "--"),
        ("total_bits_per_character", "Total: content + codebooks + metadata", "#D55E00", "^", "-."),
    ]
    for field, label, color, marker, linestyle in series:
        ax.plot(
            orders, [row[field] for row in results],
            label=label, color=color, marker=marker,
            linestyle=linestyle, linewidth=1.6, markersize=5,
        )

    ax.set_xlabel("Context length k (characters)")
    ax.set_ylabel("Bits per character")
    ax.set_title("Entropy and Huffman compression cost")
    ax.set_xticks(orders)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(output_dir / "compression_summary.png", dpi=300)
    fig.savefig(output_dir / "compression_summary.pdf")
    plt.close(fig)


# 5. 保存数据、绘图并输出最佳阶数；分析文字由报告作者完成。
def main():
    base_dir = Path(__file__).resolve().parent
    results = collect_results(base_dir)
    output_dir = base_dir / "task4_results"
    output_dir.mkdir(exist_ok=True)

    with (output_dir / "compression_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_results(results, output_dir)

    print("k     H_k       Content     Total  (bits/character)")
    for row in results:
        print(
            f"{row['k']}  {row['entropy_bits_per_character']:10.6f}"
            f"  {row['content_bits_per_character']:10.6f}"
            f"  {row['total_bits_per_character']:10.6f}"
        )
    for field, label in [
        ("content_bits_per_character", "Best content rate"),
        ("total_bits_per_character", "Best total rate"),
    ]:
        best = min(results, key=lambda row: row[field])
        print(f"{label}: k={best['k']}, {best[field]:.6f} bits/character")
    print("Costs use actual file sizes, including the existing JSON formatting.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
