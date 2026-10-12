"""Task 3: conditional Huffman coding for k=1,...,max_k."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from huffman import build_huffman_tree, build_codebook, rebuild_tree_from_codebook


# 1. 每种上下文分别统计后继字符，再分别建立 Huffman 码表。
def build_conditional_codebooks(text, k):
    if not 1 <= k < len(text):
        raise ValueError("k 必须满足 1 <= k < 文本字符数")
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        context = text[i - k:i]
        counts[context][text[i]] += 1

    return {
        context: build_codebook(build_huffman_tree(next_counts))
        for context, next_counts in counts.items()
    }


# 2. 根据前 k 个字符选择码表，将后续字符的编码打包为 bytes。
def encode_conditional(text, k, codebooks):
    output = bytearray()
    current_byte = 0
    bits_in_current_byte = 0
    valid_bit_length = 0

    for i in range(k, len(text)):
        context = text[i - k:i]
        code = codebooks[context][text[i]]
        for bit in code:
            current_byte = (current_byte << 1) | int(bit)
            bits_in_current_byte += 1
            valid_bit_length += 1
            if bits_in_current_byte == 8:
                output.append(current_byte)
                current_byte = 0
                bits_in_current_byte = 0

    padding_bits = (8 - bits_in_current_byte) % 8
    if bits_in_current_byte:
        output.append(current_byte << padding_bits)
    return bytes(output), valid_bit_length, padding_bits


# 3. 只依赖压缩内容、码表、元数据解码，不读取原文。
def decode_conditional(content, codebooks, metadata):
    k = metadata["k"]
    prefix_length = metadata["prefix_byte_length"]
    prefix = content[:prefix_length].decode("utf-8", errors="strict")
    packed_data = content[prefix_length:]
    valid_bit_length = metadata["valid_bit_length"]
    character_count = metadata["character_count"]

    if len(prefix) != k or character_count <= k:
        raise ValueError("初始字符或字符总数不正确")
    if len(packed_data) * 8 != valid_bit_length + metadata["padding_bits"]:
        raise ValueError("压缩内容长度与元数据不一致")

    trees = {
        context: rebuild_tree_from_codebook(codebook)
        for context, codebook in codebooks.items()
    }
    decoded = list(prefix)
    context = prefix
    bit_position = 0

    while len(decoded) < character_count:
        if context not in trees:
            raise ValueError(f"遇到未见上下文：{context!r}")
        node = trees[context]
        while not node.is_leaf():
            if bit_position >= valid_bit_length:
                raise ValueError("码流提前结束，无法恢复全部字符")
            byte = packed_data[bit_position // 8]
            shift = 7 - bit_position % 8
            bit = (byte >> shift) & 1
            bit_position += 1
            node = node.left if bit == 0 else node.right
            if node is None:
                raise ValueError("码流与码表不匹配")
        decoded.append(node.char)
        context = (context + node.char)[-k:]

    if bit_position != valid_bit_length:
        raise ValueError("恢复全部字符后仍有未消费的有效编码位")
    return "".join(decoded)


# 4. 使用紧凑 JSON 保存码表及元数据，大小按实际文件统计。
def write_json(path, value):
    data = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    path.write_bytes(data.encode("utf-8"))


def run_order(text, raw_bytes, k, output_dir):
    codebooks = build_conditional_codebooks(text, k)
    packed_data, valid_bit_length, padding_bits = encode_conditional(text, k, codebooks)

    # 前 k 个字符以 UTF-8 原样存入正文文件开头，之后才是 Huffman 码流。
    prefix_bytes = text[:k].encode("utf-8")
    metadata = {
        "k": k,
        "character_count": len(text),
        "prefix_byte_length": len(prefix_bytes),
        "valid_bit_length": valid_bit_length,
        "padding_bits": padding_bits,
        "source_encoding": "gb18030",
    }
    content_path = output_dir / f"k{k}.huff"
    codebook_path = output_dir / f"k{k}_codebooks.json"
    metadata_path = output_dir / f"k{k}_metadata.json"
    content_path.write_bytes(prefix_bytes + packed_data)
    write_json(codebook_path, codebooks)
    write_json(metadata_path, metadata)

    # 从磁盘重读三项压缩产物，避免验证依赖内存中的原始模型。
    del codebooks
    loaded_codebooks = json.loads(codebook_path.read_text(encoding="utf-8"))
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    decoded = decode_conditional(content_path.read_bytes(), loaded_codebooks, loaded_metadata)
    assert decoded == text, "解码字符与原文不一致"
    assert decoded.encode(loaded_metadata["source_encoding"]) == raw_bytes, "原始字节不一致"

    # 正文包括初始字符和末尾补零；总大小还包括所有解码必需的文件。
    content_bytes = content_path.stat().st_size
    codebook_bytes = codebook_path.stat().st_size
    metadata_bytes = metadata_path.stat().st_size
    total_bytes = content_bytes + codebook_bytes + metadata_bytes
    return {
        "k": k,
        "context_count": len(loaded_codebooks),
        "character_count": len(text),
        "valid_bit_length": valid_bit_length,
        "padding_bits": padding_bits,
        "prefix_bytes": len(prefix_bytes),
        "encoded_content_bytes": content_bytes,
        "codebook_bytes": codebook_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": total_bytes,
        "content_bits_per_character": content_bytes * 8 / len(text),
        "total_bits_per_character": total_bytes * 8 / len(text),
        "lossless_verified": True,
    }


# 5. 按 Task 2 的阶数逐一实验，并保存汇总 CSV。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=5, help="最大上下文长度，默认 5")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    raw_bytes = (base_dir / "GameOfThrones.txt").read_bytes()
    text = raw_bytes.decode("gb18030", errors="strict")
    if not 1 <= args.max_k < len(text):
        parser.error("--max-k 必须满足 1 <= max_k < 文本字符数")
    output_dir = base_dir / "task3_results"
    output_dir.mkdir(exist_ok=True)

    results = []
    for k in range(1, args.max_k + 1):
        row = run_order(text, raw_bytes, k, output_dir)
        results.append(row)
        print(
            f"k={k}: 正文={row['encoded_content_bytes']} B, "
            f"码表={row['codebook_bytes']} B, 元数据={row['metadata_bytes']} B, "
            f"总计={row['total_bytes']} B, 无损验证通过",
            flush=True,
        )

    with (output_dir / "conditional_huffman.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已保存至：{output_dir}")


if __name__ == "__main__":
    main()
