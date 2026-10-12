import heapq
from itertools import count


class HuffmanNode:
    def __init__(self, char=None, left=None, right=None):
        self.char = char
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(frequencies):
    heap = []
    sequence = count()  # 当频数相同时，用于稳定比较

    # 每个字符最初都是一棵只有一个叶节点的树
    for char, frequency in frequencies.items():
        node = HuffmanNode(char=char)
        heapq.heappush(
            heap,
            (frequency, next(sequence), node)
        )

    if not heap:
        raise ValueError("文本为空，不能构造 Huffman 树")

    # 每次合并频数最小的两棵树
    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)

        parent = HuffmanNode(
            left=node1,
            right=node2
        )

        heapq.heappush(
            heap,
            (freq1 + freq2, next(sequence), parent)
        )

    return heap[0][2]


def build_codebook(root):
    codebook = {}

    def traverse(node, prefix):
        if node.is_leaf():
            # 处理文本只有一种字符的特殊情况
            codebook[node.char] = prefix or "0"
            return

        traverse(node.left, prefix + "0")
        traverse(node.right, prefix + "1")

    traverse(root, "")
    return codebook

def pack_huffman_bits(text, codebook):
        """
        将 Huffman 码流打包成 bytes。
        返回：
            packed_data：压缩后的字节
            valid_bit_length：有效编码位数
            padding_bits：末尾补零数量
        """
        output = bytearray()

        current_byte = 0
        bits_in_current_byte = 0
        valid_bit_length = 0

        for char in text:
            code = codebook[char]

            for bit in code:
                current_byte = (current_byte << 1) | int(bit)
                bits_in_current_byte += 1
                valid_bit_length += 1

                # 集齐 8 位，保存为一个字节
                if bits_in_current_byte == 8:
                    output.append(current_byte)
                    current_byte = 0
                    bits_in_current_byte = 0

        # 最后不足 8 位时，在右侧补零
        if bits_in_current_byte > 0:
            padding_bits = 8 - bits_in_current_byte
            current_byte <<= padding_bits
            output.append(current_byte)
        else:
            padding_bits = 0

        return bytes(output), valid_bit_length, padding_bits

def rebuild_tree_from_codebook(codebook):
    root = HuffmanNode()

    for char, code in codebook.items():
        node = root

        for bit in code:
            if bit == "0":
                if node.left is None:
                    node.left = HuffmanNode()
                node = node.left
            else:
                if node.right is None:
                    node.right = HuffmanNode()
                node = node.right

        node.char = char

    return root

def decode_huffman(packed_data, root, valid_bit_length):
    decoded_characters = []
    node = root
    processed_bits = 0

    for byte in packed_data:
        # 从最高位向最低位读取
        for shift in range(7, -1, -1):
            if processed_bits >= valid_bit_length:
                break

            bit = (byte >> shift) & 1
            processed_bits += 1

            if bit == 0:
                node = node.left
            else:
                node = node.right

            if node is None:
                raise ValueError("压缩数据或 Huffman 码表损坏")

            if node.is_leaf():
                decoded_characters.append(node.char)
                node = root

    if processed_bits != valid_bit_length:
        raise ValueError("压缩文件中的有效位数不足")

    if node is not root:
        raise ValueError("压缩数据在一个不完整码字处结束")

    return "".join(decoded_characters)