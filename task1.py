from pathlib import Path

from collections import Counter
from math import log2

import csv
import json

from huffman import (
    build_huffman_tree,
    build_codebook,
    pack_huffman_bits,
    rebuild_tree_from_codebook,
    decode_huffman,
)

base_dir = Path(__file__).resolve().parent
file_path = base_dir / "GameOfThrones.txt"
output_dir = base_dir / "task1_results"
output_dir.mkdir(exist_ok=True)

raw_bytes = file_path.read_bytes() # 读取原始字节，返回 bytes

text = raw_bytes.decode("gb18030", errors="strict") # 按正确编码解码，返回 str

counts = Counter(text)
total = len(text)

root = build_huffman_tree(counts)
codebook = build_codebook(root)

for char in sorted(
    codebook,
    key=lambda c: (-counts[c], ord(c))
):
    print(
        f"{repr(char):10s}"
        f" 次数={counts[char]:8d}"
        f" 编码={codebook[char]}"
        f" 码长={len(codebook[char])}"
    )

h0 = -sum(
    (count / total) * log2(count / total)
    for count in counts.values()
)

print("原始字节数：", len(raw_bytes))
print("字符数：", total)
print("字符种类数：", len(counts))
print("零阶熵：", h0, "bit/字符")

(output_dir / "GameOfThrones_utf8.txt").write_text(
    text,
    encoding="utf-8",
    newline=""
)

with open(
    output_dir / "character_frequency.csv",
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow(["字符", "Unicode", "次数", "频率"])

    for char, count in counts.most_common():
        writer.writerow([
            repr(char),
            f"U+{ord(char):04X}",
            count,
            f"{count / total:.10f}",
        ])

# ============================================================
# 将 Huffman 码打包成二进制文件
# ============================================================

packed_data, valid_bit_length, padding_bits = pack_huffman_bits(
    text,
    codebook
)

compressed_path = output_dir / "GameOfThrones.huff"
compressed_path.write_bytes(packed_data)

print("\n压缩文件已生成：", compressed_path)
print("有效编码位数：", valid_bit_length)
print("压缩文件字节数：", len(packed_data))
print("末尾补零位数：", padding_bits)

# 保存解码所需的码表和元数据
metadata = {
    "character_count": len(text),
    "valid_bit_length": valid_bit_length,
    "padding_bits": padding_bits,
    "codebook": codebook,
}

metadata_path = output_dir / "huffman_metadata.json"

with metadata_path.open(
    "w",
    encoding="utf-8",
    newline=""
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=True,
        indent=2
    )

print("元数据文件已生成：", metadata_path)

# 从磁盘重新读取压缩文件和元数据
loaded_packed_data = compressed_path.read_bytes()

with metadata_path.open(
    "r",
    encoding="utf-8"
) as file:
    loaded_metadata = json.load(file)

loaded_codebook = loaded_metadata["codebook"]
loaded_bit_length = loaded_metadata["valid_bit_length"]
expected_character_count = loaded_metadata["character_count"]

# 根据保存的码表重建解码树
decode_root = rebuild_tree_from_codebook(loaded_codebook)

# 解码二进制数据
decoded_text = decode_huffman(
    loaded_packed_data,
    decode_root,
    loaded_bit_length
)

# 无损验证
assert len(decoded_text) == expected_character_count
assert decoded_text == text
assert decoded_text.encode("gb18030") == raw_bytes

print("\n无损解码验证成功！")
print("解码字符数：", len(decoded_text))
print("解码文本与原始文本完全一致。")
print("重新编码后的字节与原始文件完全一致。")

decoded_path = output_dir / "GameOfThrones_decoded.txt"

decoded_path.write_text(
    decoded_text,
    encoding="utf-8",
    newline=""
)

print("解码结果已保存：", decoded_path)

effective_rate = valid_bit_length / len(text)
physical_rate = len(packed_data) * 8 / len(text)
entropy_gap = effective_rate - h0

print("\n最终压缩统计：")
print(f"原始字符数：{len(text)}")
print(f"有效编码位数：{valid_bit_length}")
print(f"压缩文件大小：{len(packed_data)} 字节")
print(f"有效 Huffman 码率：{effective_rate:.12f} bit/字符")
print(f"包含补零的文件码率：{physical_rate:.12f} bit/字符")
print(f"零阶熵 H0：{h0:.12f} bit/字符")
print(f"码率与 H0 的差：{entropy_gap:.12f} bit/字符")
