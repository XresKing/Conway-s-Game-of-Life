"""外部图案导入：把常见的生命游戏文件格式解析成相对偏移 (dy, dx)。

支持三种社区通用格式，可自动识别：

* **RLE** （``.rle``，最常见，LifeWiki 默认）—— ``x = .., y = ..`` 头 + 行程编码体；
* **Plaintext** （``.cells``）—— ``.`` 表示死、``O`` 表示活，``!`` 起注释行；
* **Life 1.06** （``#Life 1.06``）—— 每行一个 ``x y`` 活细胞绝对坐标。

所有解析函数都返回 *以图案几何中心为原点* 的偏移列表 ``[(dy, dx), ...]``，
这样无论棋盘多大、中心在哪，都能用 ``presets.offsets_to_coords`` 居中落子。
"""

from __future__ import annotations

import re


class PatternParseError(ValueError):
    """图案文本无法解析时抛出，消息对用户友好。"""


# ---------------------------------------------------------------------------
# 居中工具
# ---------------------------------------------------------------------------
def _center_cells(cells):
    """把绝对 (row, col) 列表归一到以几何中心为原点的整数偏移 (dy, dx)。"""
    if not cells:
        return []
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    # 单一整数平移（按外接框中点取整），保证所有活细胞保持互不重合
    sy = (min(rows) + max(rows)) // 2
    sx = (min(cols) + max(cols)) // 2
    out = {(r - sy, c - sx) for r, c in cells}
    return sorted(out)


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------
_RLE_HEADER = re.compile(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", re.IGNORECASE)
_RLE_TOKEN = re.compile(r"(\d*)([bo$!])")


def parse_rle(text: str):
    """解析 RLE 文本，返回居中后的相对偏移 (dy, dx)。"""
    header_seen = False
    body = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue  # 注释 / 元信息行（#C #N #O #r 等）
        if not header_seen and _RLE_HEADER.search(line):
            header_seen = True
            # 头行末尾可能跟着 "rule = B3/S23"，规则交给主程序，这里忽略
            continue
        body.append(line)

    if not body:
        raise PatternParseError("RLE 中没有找到图案数据（只有注释或空行）。")

    stream = "".join(body)
    cells = []
    row = col = 0
    terminated = False
    for count_str, tag in _RLE_TOKEN.findall(stream):
        count = int(count_str) if count_str else 1
        if tag == "b":          # 死细胞：跳过
            col += count
        elif tag == "o":        # 活细胞
            for _ in range(count):
                cells.append((row, col))
                col += 1
        elif tag == "$":        # 换行（可一次跳多行）
            row += count
            col = 0
        elif tag == "!":        # 结束
            terminated = True
            break

    if not terminated and not cells:
        raise PatternParseError("RLE 内容无法识别为有效的行程编码。")
    return _center_cells(cells)


# ---------------------------------------------------------------------------
# Plaintext (.cells)
# ---------------------------------------------------------------------------
def parse_plaintext(text: str):
    """解析 Plaintext (.cells)：``.``=死、``O``/``*``/``#``=活，``!`` 起注释。"""
    cells = []
    row = 0
    saw_grid = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("!"):
            continue  # 注释行
        if line.strip() == "":
            # 空行也算一行（保留图案内部的纵向空隙），但不在末尾累计
            row += 1
            continue
        saw_grid = True
        for col, ch in enumerate(line):
            if ch in "O*#":
                cells.append((row, col))
        row += 1

    if not saw_grid:
        raise PatternParseError("Plaintext 中没有找到任何活细胞行。")
    return _center_cells(cells)


# ---------------------------------------------------------------------------
# Life 1.06
# ---------------------------------------------------------------------------
def parse_life106(text: str):
    """解析 Life 1.06：``#Life 1.06`` 头 + 每行一个 ``x y`` 活细胞坐标。"""
    cells = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise PatternParseError(f"Life 1.06 坐标行格式错误: {raw!r}")
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise PatternParseError(f"Life 1.06 坐标必须为整数: {raw!r}") from exc
        cells.append((y, x))  # 文件是 (x, y)，内部用 (row=y, col=x)
    if not cells:
        raise PatternParseError("Life 1.06 中没有任何坐标。")
    return _center_cells(cells)


# ---------------------------------------------------------------------------
# 自动识别 + 统一入口
# ---------------------------------------------------------------------------
def parse_pattern(text: str):
    """自动识别格式并解析，返回 (offsets, format_name)。

    offsets 为以几何中心为原点的相对偏移 [(dy, dx), ...]，可直接喂给
    ``presets.offsets_to_coords``。识别失败时抛 :class:`PatternParseError`。
    """
    if not text or not text.strip():
        raise PatternParseError("内容为空。")

    head = text.lstrip()
    lowered = head.lower()

    if lowered.startswith("#life 1.06") or lowered.startswith("#life 1.05"):
        return parse_life106(text), "Life 1.06"

    # RLE：有 x = .., y = .. 头，或正文出现 b/o/$ 行程符号
    if _RLE_HEADER.search(text) or re.search(r"\d*[bo]\d*\$", text):
        return parse_rle(text), "RLE"

    # 其余按 Plaintext 处理（.cells / 纯 ASCII 网格）
    return parse_plaintext(text), "Plaintext"
