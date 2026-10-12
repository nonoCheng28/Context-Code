# 信息论基础 Project 1

以《权力的游戏》文本为数据，研究字符熵、条件熵与 Huffman 无损压缩，并通过跨小说实验比较概率模型的迁移表现。

## 实验内容

| 脚本 | 功能 |
| --- | --- |
| `task1.py` | 统计字符频率与零阶熵，完成 Huffman 编码、解码及无损验证 |
| `task2.py` | 计算上下文长度为 0–5 的经验条件熵并绘图 |
| `task3.py` | 实现上下文长度为 1–5 的条件 Huffman 编码及无损验证 |
| `task4.py` | 汇总熵、正文码率及含码表和元数据的总存储成本 |
| `huffman.py` | Huffman 树、码表及编解码工具 |
| `cross_novel.py` | 比较《权力的游戏》与《了不起的盖茨比》的交叉熵和条件 KL 散度 |

## 快速运行

在本 README 所在目录中，使用 Python 3 执行：

```bash
python -m pip install matplotlib
python task1.py
python task2.py --max-k 5
python task3.py --max-k 5
python task4.py
```

Task 2、3 的最大上下文长度须保持一致，Task 4 依赖前三项的输出。结果分别保存至 `task1_results/`–`task4_results/`，包括 CSV、压缩文件、码表及图像；重新运行会更新对应结果。仅 Task 2、4 的绘图需要 Matplotlib，其余脚本使用 Python 标准库。

## 跨小说扩展实验

仓库已包含预处理输入，可直接运行：

```bash
python cross_novel.py --orders 2 3 --alpha 0.1
```

结果保存在 `cross_novel_results/`。如需重建输入，先执行 `python prepare_cross_novel.py`；缺少来源归档时，脚本会联网下载固定版本的源文本。

数据来源与预处理规则见 [数据说明](cross_novel_data/README.md)，指标定义及结果见 [实验说明](cross_novel_results/README.md)。扩展实验计算的是理论交叉熵，不生成实际 Huffman 压缩文件。

## 数据口径

Task 1–4 按 GB18030 解码根目录的 `GameOfThrones.txt`，保留大小写、标点、空白和原始换行。跨小说实验使用 `cross_novel_data/` 中统一预处理后的 UTF-8 文本。熵与码率以 bit/字符计；比较压缩效果时，应区分正文成本和包含码表、元数据的总成本。
