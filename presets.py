"""康威生命游戏的预设图案。

坐标使用 (dy, dx) 相对偏移；`make_pattern` 以某个中心点展开成绝对坐标。
预设按类别组织（静物 / 振荡器 / 飞船 / 玛士撒拉 / 枪），既供下拉框选择，
也供文档页分门别类地展示。
"""

import numpy as np


def make_pattern(relative_coords, center=(50, 50), dtype=np.int32):
    """把相对偏移 (dy, dx) 列表以 center 为中心展开成绝对坐标 (N, 2)。"""
    return np.array(
        [[center[0] + dy, center[1] + dx] for dy, dx in relative_coords],
        dtype=dtype,
    )


def _centered(rows):
    """把一个 ``"....OO.."`` 形式的 ASCII 行列表转成以图案几何中心为原点的
    相对偏移 (dy, dx) 列表（``O``/``*``/``#`` 视为活细胞）。

    这样写图案更直观、易于校对，且无论图案多大都会自动居中。
    """
    cells = [
        (r, c)
        for r, line in enumerate(rows)
        for c, ch in enumerate(line)
        if ch in "O*#"
    ]
    if not cells:
        return []
    rows_i = [r for r, _ in cells]
    cols_i = [c for _, c in cells]
    # 单一整数平移（按外接框中点取整），避免逐格 round 把相邻格压到一起
    sy = (min(rows_i) + max(rows_i)) // 2
    sx = (min(cols_i) + max(cols_i)) // 2
    return [(r - sy, c - sx) for r, c in cells]


# ---------------------------------------------------------------------------
# 2D 预设：按类别给出相对偏移（不含中心），便于在任意中心/边界下重新展开
# ---------------------------------------------------------------------------
PATTERN_CATEGORIES = {
    "静物 Still life": {
        "block": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "beehive": _centered([".OO.", "O..O", ".OO."]),
        "loaf": _centered([".OO.", "O..O", ".O.O", "..O."]),
        "boat": _centered(["OO.", "O.O", ".O."]),
        "tub": _centered([".O.", "O.O", ".O."]),
        "pond": _centered([".OO.", "O..O", "O..O", ".OO."]),
        "ship": _centered(["OO.", "O.O", ".OO"]),
    },
    "振荡器 Oscillator": {
        "blinker": [(0, -1), (0, 0), (0, 1)],
        "toad": _centered([".OOO", "OOO."]),
        "beacon": _centered(["OO..", "OO..", "..OO", "..OO"]),
        "clock": _centered(["..O.", "O.O.", ".O.O", ".O.."]),
        "pulsar": _centered([
            "..OOO...OOO..",
            "............",
            "O....O.O....O",
            "O....O.O....O",
            "O....O.O....O",
            "..OOO...OOO..",
            "............",
            "..OOO...OOO..",
            "O....O.O....O",
            "O....O.O....O",
            "O....O.O....O",
            "............",
            "..OOO...OOO..",
        ]),
        "pentadecathlon": _centered([
            "..O....O..",
            "OO.OOOO.OO",
            "..O....O..",
        ]),
        "figure_eight": _centered([
            "OOO...",
            "OOO...",
            "OOO...",
            "...OOO",
            "...OOO",
            "...OOO",
        ]),
    },
    "飞船 Spaceship": {
        "glider": [(-1, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
        "lwss": _centered([
            "O..O.",
            "....O",
            "O...O",
            ".OOOO",
        ]),
        "mwss": _centered([
            "...O..",
            ".O...O",
            "O.....",
            "O....O",
            "OOOOO.",
        ]),
        "hwss": _centered([
            "...OO...",
            ".O....O.",
            "O.......",
            "O......O",
            "OOOOOO..",
        ]),
        "loafer": _centered([
            ".OO..O.OOO",
            "O..O..OO..",
            ".O.O......",
            "..O.......",
            "........OO",
            ".......O.O",
            "........O.",
        ]),
    },
    "玛士撒拉 Methuselah": {
        # 小种子、长寿命：演化数百代后才稳定，最具观赏性
        "r_pentomino": _centered([".OO", "OO.", ".O."]),
        "diehard": _centered([
            "......O.",
            "OO......",
            ".O...OOO",
        ]),
        "acorn": _centered([
            ".O.....",
            "...O...",
            "OO..OOO",
        ]),
        "b_heptomino": _centered([
            "O..",
            "OOO",
            "O.O",
        ]),
    },
    "枪 / 复杂 Gun & complex": {
        "gosper_glider_gun": _centered([
            "........................O...........",
            "......................O.O...........",
            "............OO......OO............OO",
            "...........O...O....OO............OO",
            "OO........O.....O...OO..............",
            "OO........O...O.OO....O.O...........",
            "..........O.....O.......O...........",
            "...........O...O....................",
            "............OO......................",
        ]),
        "simkin_glider_gun": _centered([
            "OO.....OO........................",
            "OO.....OO........................",
            ".................................",
            "....OO...........................",
            "....OO...........................",
            ".................................",
            ".................................",
            ".................................",
            ".................................",
            "......................OO.OO......",
            ".....................O.....O.....",
            ".....................O......O..OO",
            ".....................OOO...O...OO",
            "..........................O......",
            ".................................",
            ".................................",
            ".................................",
            "....................OO...........",
            "....................O............",
            ".....................OOO.........",
            ".......................O.........",
        ]),
    },
}

# 把分类摊平成 name -> offsets，兼容旧接口
PATTERN_OFFSETS = {
    name: offsets
    for group in PATTERN_CATEGORIES.values()
    for name, offsets in group.items()
}

# 兼容旧接口：以默认中心 (50, 50) 展开的绝对坐标字典
PATTERNS = {name: make_pattern(offsets) for name, offsets in PATTERN_OFFSETS.items()}

PATTERN_NAMES = list(PATTERN_OFFSETS)


def pattern_coords(name, center=(50, 50), dtype=np.int32):
    """返回指定预设以 center 为中心展开的绝对坐标 (N, 2) int32。"""
    if name not in PATTERN_OFFSETS:
        raise KeyError(f"未知预设: {name!r}，可选: {PATTERN_NAMES}")
    return make_pattern(PATTERN_OFFSETS[name], center=center, dtype=dtype)


def offsets_to_coords(offsets, center, dtype=np.int32):
    """把任意 (dy, dx) 相对偏移列表以 center 为中心展开成绝对坐标 (N, 2)。

    供"外部数据导入"复用：解析出的图案先归一到以几何中心为原点的偏移，
    再用此函数落到当前棋盘中心。
    """
    if len(offsets) == 0:
        return np.zeros((0, 2), dtype=dtype)
    return make_pattern(offsets, center=center, dtype=dtype)


# ----------------------------------------------------------------------------
# 3D 预设：相对偏移 (dz, dy, dx)，以某个三维中心展开成绝对坐标 (N, 3)
# 规则按 Bays 5766（B6/S567），种子需要一定厚度才能持续演化。
# ----------------------------------------------------------------------------
PATTERN_OFFSETS_3D = {
    "cube": [  # 2x2x2 实心立方体（稳定块）
        (z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)
    ],

    "box": [  # 3x3x3 空心盒（外壳）
        (z, y, x)
        for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
        if abs(z) == 1 or abs(y) == 1 or abs(x) == 1
    ],

    "blinker3d": [  # 沿三轴的小十字
        (0, 0, 0),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ],

    "solid_cube": [  # 3x3x3 实心立方体（活跃种子）
        (z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
    ],

    "plus3d": [  # 三维加号：中心 + 六个面方向各延伸一格
        (0, 0, 0),
        (2, 0, 0), (-2, 0, 0),
        (0, 2, 0), (0, -2, 0),
        (0, 0, 2), (0, 0, -2),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ],

    "shell4": [  # 4x4x4 实心立方体：体量更大、演化更剧烈
        (z, y, x)
        for z in (-2, -1, 0, 1) for y in (-2, -1, 0, 1) for x in (-2, -1, 0, 1)
    ],
}

PATTERN_NAMES_3D = list(PATTERN_OFFSETS_3D)


def pattern_coords_3d(name, center=(25, 25, 25), dtype=np.int32):
    """返回指定 3D 预设以 center 为中心展开的绝对坐标 (N, 3) int32，列序 (z, y, x)。"""
    if name not in PATTERN_OFFSETS_3D:
        raise KeyError(f"未知 3D 预设: {name!r}，可选: {PATTERN_NAMES_3D}")
    cz, cy, cx = center
    return np.array(
        [[cz + dz, cy + dy, cx + dx] for dz, dy, dx in PATTERN_OFFSETS_3D[name]],
        dtype=dtype,
    )
