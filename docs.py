"""文档页 ``/docs``：图文并茂地介绍康威生命游戏。

页面里的动画并非贴图，而是用一个微型纯 Python 生命引擎 (`_step`) 真实演化
若干代，再把每一帧编码成 SVG，靠 SMIL 的 ``<animate>`` 离散切换可见性循环播放——
因此"滑翔机滑行""脉冲星脉动""高斯帕枪喷射滑翔机"都是按 B3/S23 规则算出来的真迹。
配色沿用 DESIGN.md 的 Apple 视觉语言（Action Blue #0066cc、发光蓝 #2997ff）。
"""

from __future__ import annotations

from collections import Counter

from nicegui import ui

from presets import PATTERN_CATEGORIES, PATTERN_OFFSETS

# 设计系统取色
INK = "#1d1d1f"
MUTED = "#6e6e73"
ACTION = "#0066cc"
GLOW = "#2997ff"
HAIRLINE = "#e0e0e0"
CANVAS = "#f5f5f7"


# ---------------------------------------------------------------------------
# 微型生命引擎（仅供文档插图，体量小，纯 Python 足够快）
# ---------------------------------------------------------------------------
def _step(alive, rows, cols, wrap, born=(3,), survive=(2, 3)):
    """演化一代，返回新的活细胞集合 {(y, x)}。wrap=True 时网格为环面。"""
    born, survive = set(born), set(survive)
    counts = Counter()
    for (y, x) in alive:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if wrap:
                    ny, nx = ny % rows, nx % cols
                elif not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                counts[(ny, nx)] += 1
    nxt = {c for c in alive if counts.get(c, 0) in survive}
    nxt |= {c for c, n in counts.items() if n in born and c not in alive}
    return nxt


def _simulate(initial, rows, cols, frames, wrap, born=(3,), survive=(2, 3)):
    """从 initial 演化 frames 帧，返回每帧的活细胞集合列表（含第 0 帧）。"""
    alive = set(initial)
    out = [set(alive)]
    for _ in range(frames - 1):
        alive = _step(alive, rows, cols, wrap, born, survive)
        out.append(set(alive))
    return out


# ---------------------------------------------------------------------------
# SVG 绘制
# ---------------------------------------------------------------------------
def _normalize(offsets, pad=1):
    """把以中心为原点的偏移搬到非负网格，返回 (cells, rows, cols)。"""
    ys = [y for y, _ in offsets]
    xs = [x for _, x in offsets]
    miny, minx = min(ys), min(xs)
    cells = [(y - miny + pad, x - minx + pad) for y, x in offsets]
    rows = max(ys) - miny + 1 + 2 * pad
    cols = max(xs) - minx + 1 + 2 * pad
    return cells, rows, cols


def _grid_backdrop(rows, cols, cell):
    """深色棋盘底 + 细网格线。"""
    w, h = cols * cell, rows * cell
    parts = [f'<rect x="0" y="0" width="{w}" height="{h}" rx="10" fill="#0a0a0c"/>']
    for c in range(cols + 1):
        parts.append(
            f'<line x1="{c*cell}" y1="0" x2="{c*cell}" y2="{h}" '
            f'stroke="#1c1c22" stroke-width="1"/>'
        )
    for r in range(rows + 1):
        parts.append(
            f'<line x1="0" y1="{r*cell}" x2="{w}" y2="{r*cell}" '
            f'stroke="#1c1c22" stroke-width="1"/>'
        )
    return "".join(parts), w, h


def _cells_rects(cells, cell, color=GLOW, inset=1.4, rx=2.4):
    out = []
    s = cell - 2 * inset
    for (y, x) in cells:
        out.append(
            f'<rect x="{x*cell+inset:.1f}" y="{y*cell+inset:.1f}" '
            f'width="{s:.1f}" height="{s:.1f}" rx="{rx}" fill="{color}"/>'
        )
    return "".join(out)


def static_svg(offsets, cell=18, color=GLOW, pad=1):
    """把一个图案画成静态 SVG（深色棋盘 + 发光蓝方块）。空图案画成 3×3 空棋盘。"""
    if len(offsets) == 0:
        backdrop, w, h = _grid_backdrop(3, 3, cell)
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">'
            f"{backdrop}</svg>"
        )
    cells, rows, cols = _normalize(offsets, pad=pad)
    backdrop, w, h = _grid_backdrop(rows, cols, cell)
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">'
        f'{backdrop}{_cells_rects(cells, cell, color)}</svg>'
    )


def _anim_keys(i, frames):
    """第 i 帧（共 frames 帧）的离散可见性 keyTimes / values。"""
    a, b = i / frames, (i + 1) / frames
    if i == 0:
        return f"0;{b:.4f};1", "1;0;0"
    if i == frames - 1:
        return f"0;{a:.4f};1", "0;1;1"
    return f"0;{a:.4f};{b:.4f};1", "0;1;0;0"


def animated_svg(initial, rows, cols, frames, wrap=True, cell=16,
                 dt=0.16, color=GLOW, born=(3,), survive=(2, 3)):
    """真实演化 frames 帧并编码成循环播放的 SVG 动画。

    initial 为以网格左上为原点的 (y, x) 活细胞；用 SMIL 离散切换每帧 `<g>` 的
    opacity，所以无需 JS、离线可放，且帧内容就是规则算出的真演化。
    """
    history = _simulate(initial, rows, cols, frames, wrap, born, survive)
    backdrop, w, h = _grid_backdrop(rows, cols, cell)
    total = frames * dt
    groups = []
    for i, alive in enumerate(history):
        keytimes, values = _anim_keys(i, frames)
        rects = _cells_rects(sorted(alive), cell, color)
        groups.append(
            f'<g opacity="{1 if i == 0 else 0}">'
            f'<animate attributeName="opacity" dur="{total:.2f}s" '
            f'repeatCount="indefinite" calcMode="discrete" '
            f'keyTimes="{keytimes}" values="{values}"/>{rects}</g>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;">'
        f'{backdrop}{"".join(groups)}</svg>'
    )


def _rule_panel(title, before, after, verdict, ok):
    """规则四联图中的一格：演化前 → 演化后 + 判定。"""
    badge = "#34c759" if ok else "#ff453a"
    arrow = (
        f'<span style="color:{MUTED};font-size:20px;margin:0 6px;">→</span>'
    )
    return (
        f'<div style="display:flex;flex-direction:column;gap:8px;align-items:center;'
        f'background:#fff;border:1px solid {HAIRLINE};border-radius:14px;padding:16px;">'
        f'<div style="display:flex;align-items:center;">{static_svg(before, cell=15)}'
        f'{arrow}{static_svg(after, cell=15)}</div>'
        f'<div style="font-weight:600;color:{INK};font-size:14px;">{title}</div>'
        f'<div style="color:{badge};font-size:12.5px;font-weight:600;">{verdict}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# 页面骨架
# ---------------------------------------------------------------------------
def _section_title(kicker, title, lead=None):
    ui.html(
        f'<div style="margin:0 0 4px;color:{ACTION};font-size:13px;font-weight:600;'
        f'letter-spacing:0.4px;text-transform:uppercase;">{kicker}</div>'
        f'<div style="font-size:32px;font-weight:600;letter-spacing:-0.4px;'
        f'color:{INK};line-height:1.1;">{title}</div>'
        + (
            f'<div style="font-size:18px;color:{MUTED};margin-top:10px;'
            f'line-height:1.55;max-width:680px;">{lead}</div>'
            if lead
            else ""
        )
    )


def _card(content_html):
    ui.html(
        f'<div style="background:#fff;border:1px solid {HAIRLINE};border-radius:18px;'
        f'padding:26px 28px;line-height:1.65;color:{INK};font-size:16px;">'
        f"{content_html}</div>"
    )


def build_docs():
    """构建 /docs 页面正文（不含全局 head，由调用方注入）。"""
    ui.colors(primary=ACTION)

    # —— 顶部导航 ——
    with ui.row().classes("w-full items-center no-wrap").style(
        "position:sticky;top:0;z-index:50;background:rgba(245,245,247,0.8);"
        "backdrop-filter:saturate(180%) blur(20px);"
        f"border-bottom:1px solid {HAIRLINE};padding:14px 28px;"
    ):
        ui.html(
            f'<div style="font-weight:600;font-size:17px;color:{INK};">'
            f'康威生命游戏 <span style="color:{MUTED};font-weight:400;">· 文档</span></div>'
        )
        ui.space()
        ui.link("← 返回模拟器", "/").style(
            f"color:{ACTION};font-weight:600;font-size:15px;text-decoration:none;"
        )

    # —— 内容容器 ——
    with ui.column().classes("items-center w-full").style(
        f"background:{CANVAS};padding:0 20px 100px;"
    ):
        with ui.column().classes("w-full").style("max-width:880px;gap:64px;"):
            _hero()
            _what_is()
            _rules()
            _taxonomy()
            _applications()
            _import_formats()
            _references()


# ---------------------------------------------------------------------------
# 各章节
# ---------------------------------------------------------------------------
def _hero():
    glider = [(y, x) for y, x in __glider_seed()]
    anim = animated_svg(glider, rows=20, cols=22, frames=24, wrap=True,
                        cell=15, dt=0.13)
    with ui.column().classes("items-center w-full").style("padding-top:72px;gap:22px;"):
        ui.html(
            f'<div style="text-align:center;">'
            f'<div style="font-size:15px;font-weight:600;color:{ACTION};'
            f'letter-spacing:0.5px;">CONWAY\'S GAME OF LIFE</div>'
            f'<div style="font-size:54px;font-weight:600;letter-spacing:-1px;'
            f'color:{INK};line-height:1.05;margin-top:10px;">无中生有的宇宙</div>'
            f'<div style="font-size:20px;color:{MUTED};margin-top:16px;'
            f'max-width:640px;line-height:1.5;">四条极简规则，催生出滑翔机、振荡器，'
            f'乃至能自我复制、可进行通用计算的复杂结构——这就是元胞自动机的魅力。</div>'
            f"</div>"
        )
        ui.html(
            f'<div style="box-shadow:0 24px 60px -20px rgba(0,0,0,0.4);'
            f'border-radius:14px;line-height:0;">{anim}</div>'
        )
        ui.html(
            f'<div style="color:{MUTED};font-size:13px;">↑ 一只滑翔机（glider）在环面网格上'
            f'按 B3/S23 规则真实演化——每 4 代沿对角线平移一格</div>'
        )


def _what_is():
    _section_title(
        "ORIGIN", "它是什么？",
        "1970 年，英国数学家 John Horton Conway 发明了这个"
        "“零玩家游戏”。它发表于 Martin Gardner 在《科学美国人》的专栏，"
        "瞬间风靡全球。",
    )
    _card(
        "<b>生命游戏</b>是一种<b>二维元胞自动机</b>：无限的方格网中，每个格子（细胞）"
        "只有“生”与“死”两种状态。下一代的状态完全由它当前周围 8 个邻居的存活数决定，"
        "无需任何随机性——因此称为<b>确定性</b>系统，也是<b>“零玩家游戏”</b>："
        "你只设定初始布局，之后宇宙自行运转。<br><br>"
        "尽管规则简单到能写在一张便签上，它却被证明是<b>图灵完备</b>的——"
        "理论上能模拟任何计算机程序。简单规则 + 局部交互 → 涌现出全局复杂性，"
        "这正是<b>复杂系统科学</b>的标志性范例。"
    )


def _rules():
    _section_title(
        "THE RULES", "四条规则，仅此而已",
        "每一代，所有细胞同时按下面的规则更新。记法 B3/S23：邻居恰为 3 时"
        "“出生(Born)”，邻居为 2 或 3 时“存活(Survive)”，其余皆死。",
    )
    # 四个最小示例（以中心为原点的偏移）
    under = ([(0, 0), (0, 1)], [])                      # 1 邻居 → 死
    survive = ([(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 0), (1, 1)])
    over = ([(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)], [(-1, 0), (0, -1), (0, 1), (1, 0)])
    born = ([(0, -1), (0, 1), (1, 0)], [(0, -1), (0, 0), (0, 1), (1, 0)])
    panels = (
        _rule_panel("人口过少", under[0], under[1], "邻居 &lt; 2 · 死亡", False)
        + _rule_panel("安居存活", survive[0], survive[1], "邻居 = 2 或 3 · 存活", True)
        + _rule_panel("人口过密", over[0], over[1], "邻居 &gt; 3 · 死亡", False)
        + _rule_panel("繁殖新生", born[0], born[1], "空格邻居 = 3 · 出生", True)
    )
    ui.html(
        f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;">'
        f"{panels}</div>"
    )
    _card(
        "改变这两个数字集合，就得到一整族“类生命”规则：例如 <b>B36/S23</b>"
        "（HighLife，会出现自我复制结构）、<b>B1/S012345678</b>（Replicator，万物皆自我复制）。"
        "本模拟器左栏的“出生/存活”输入框，正是让你随意改写这两条规则——"
        "三维模式则默认采用 Bays 的 <b>B6/S567</b>。"
    )


def _taxonomy():
    _section_title(
        "MENAGERIE", "图案动物园",
        "无数初始布局被人们发现、命名、收藏。按行为可分为四大类，"
        "它们也都内置在本模拟器的预设里——下拉即可载入。",
    )
    # 用真实动画展示每一类的代表
    blinker = [(0, 0), (0, 1), (0, 2)]
    pulsar = PATTERN_OFFSETS["pulsar"]
    lwss = PATTERN_OFFSETS["lwss"]
    cards = [
        ("静物 · Still life", "永不改变的稳定结构。方块、蜂巢、面包……邻居数恰好让每个细胞自洽。",
         _still_strip(["block", "beehive", "loaf", "boat"])),
        ("振荡器 · Oscillator", "周期性循环的图案。闪烁灯周期 2，脉冲星周期 3。",
         _anim_centered(pulsar, frames=3, dt=0.5, pad=2, wrap=False, cell=12)),
        ("飞船 · Spaceship", "一边演化一边整体平移，能在网格上“飞行”。轻量级飞船 LWSS 每 4 代移动 2 格。",
         _anim_centered(lwss, frames=16, dt=0.18, pad=3, rows_min=11, cols_min=22)),
        ("玛士撒拉 · Methuselah", "极小的种子却能搅动数百代才安定。R-五连块演化 1103 代后才稳定下来。",
         _anim_centered(PATTERN_OFFSETS["r_pentomino"], frames=40, dt=0.12,
                        pad=14, wrap=False, cell=6)),
    ]
    for title, desc, svg in cards:
        ui.html(
            f'<div style="display:flex;gap:24px;align-items:center;background:#fff;'
            f'border:1px solid {HAIRLINE};border-radius:18px;padding:22px 26px;'
            f'flex-wrap:wrap;">'
            f'<div style="flex:0 0 auto;line-height:0;">{svg}</div>'
            f'<div style="flex:1 1 240px;min-width:240px;">'
            f'<div style="font-size:20px;font-weight:600;color:{INK};">{title}</div>'
            f'<div style="color:{MUTED};font-size:15.5px;margin-top:8px;'
            f'line-height:1.55;">{desc}</div></div></div>'
        )
    # 高斯帕滑翔机枪——会议级名场面，单独大图
    gun = PATTERN_OFFSETS["gosper_glider_gun"]
    cells, _, _ = _normalize(gun, pad=2)
    rows = max(38, max(y for y, _ in cells) + 16)
    cols = max(y for _, y in cells) + 18
    gun_anim = animated_svg(cells, rows=40, cols=cols, frames=32, wrap=False,
                            cell=8, dt=0.12)
    ui.html(
        f'<div style="background:#fff;border:1px solid {HAIRLINE};border-radius:18px;'
        f'padding:24px 26px;">'
        f'<div style="font-size:20px;font-weight:600;color:{INK};">'
        f'高斯帕滑翔机枪 · Gosper Glider Gun</div>'
        f'<div style="color:{MUTED};font-size:15.5px;margin:8px 0 18px;line-height:1.55;">'
        f'1970 年由 Bill Gosper 发现的第一个“枪”——它周期性地永久喷射滑翔机，'
        f'证明了生命游戏中存在<b>无限增长</b>的图案，并为日后用滑翔机搭建逻辑电路'
        f'（与门、或门、存储器）铺平了道路。</div>'
        f'<div style="line-height:0;overflow-x:auto;">{gun_anim}</div></div>'
    )


def _applications():
    _section_title(
        "WHY IT MATTERS", "不只是好看",
        "生命游戏是横跨数学、计算机科学、哲学与艺术的思想实验。",
    )
    items = [
        ("🧮 通用计算", "用滑翔机当信号、用碰撞当逻辑门，人们在生命游戏里搭出了"
         "可工作的逻辑电路、寄存器，甚至一台能运行生命游戏的元胞计算机。它是图灵完备的。"),
        ("🌱 涌现与复杂系统", "简单局部规则催生全局复杂行为，是研究自组织、相变、"
         "人工生命的经典模型，启发了对生物形态发生与群体行为的建模。"),
        ("🔬 自我复制", "Conway 曾猜想存在能自我复制的图案；2010 年的 Gemini 飞船"
         "与 2013 年的 Linear Propagator 证实了这一点，呼应冯·诺依曼的自复制自动机。"),
        ("🎨 生成艺术", "确定性却不可预测的演化天然适合生成式视觉与音乐创作，"
         "也是教学中讲解算法、并行计算与稀疏数据结构的绝佳载体。"),
    ]
    cells = "".join(
        f'<div style="background:#fff;border:1px solid {HAIRLINE};border-radius:16px;'
        f'padding:22px 24px;">'
        f'<div style="font-size:18px;font-weight:600;color:{INK};">{t}</div>'
        f'<div style="color:{MUTED};font-size:15.5px;margin-top:8px;line-height:1.6;">'
        f"{d}</div></div>"
        for t, d in items
    )
    ui.html(
        f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;">'
        f"{cells}</div>"
    )


def _import_formats():
    _section_title(
        "BRING YOUR OWN", "导入外部图案",
        "模拟器左栏的“导入图案”支持三种社区通用格式，自动识别。"
        "你可以从 LifeWiki 等图案库下载文件粘贴进来，或直接上传。",
    )
    rle = (
        "#N Glider\n"
        "#C 最常见的格式，LifeWiki 默认\n"
        "x = 3, y = 3, rule = B3/S23\n"
        "bob$2bo$3o!"
    )
    cells = "!Name: Block\n!\nOO\nOO"
    l106 = "#Life 1.06\n0 1\n1 2\n2 0\n2 1\n2 2"
    fmts = [
        ("RLE", ".rle", "行程编码：<code>b</code>=死、<code>o</code>=活、"
         "<code>$</code>=换行、<code>!</code>=结束；头行声明尺寸与规则。", rle),
        ("Plaintext", ".cells", "纯文本网格：<code>.</code>=死、<code>O</code>=活，"
         "<code>!</code> 开头为注释。直观易手写。", cells),
        ("Life 1.06", ".lif", "每行一个活细胞的 <code>x y</code> 绝对坐标，适合稀疏大图。", l106),
    ]
    for name, ext, desc, sample in fmts:
        ui.html(
            f'<div style="background:#fff;border:1px solid {HAIRLINE};border-radius:16px;'
            f'padding:22px 24px;margin-bottom:14px;">'
            f'<div style="display:flex;align-items:baseline;gap:10px;">'
            f'<span style="font-size:18px;font-weight:600;color:{INK};">{name}</span>'
            f'<span style="font-size:13px;color:{ACTION};font-family:monospace;">{ext}</span>'
            f"</div>"
            f'<div style="color:{MUTED};font-size:15px;margin:8px 0 14px;line-height:1.55;">'
            f"{desc}</div>"
            f'<pre style="background:{CANVAS};border:1px solid {HAIRLINE};border-radius:10px;'
            f'padding:14px 16px;font-size:13px;color:{INK};overflow-x:auto;margin:0;'
            f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">'
            f"{_escape(sample)}</pre></div>"
        )


def _references():
    links = [
        ("LifeWiki — 图案百科与下载", "https://conwaylife.com/wiki/"),
        ("Golly — 高性能开源生命游戏模拟器", "https://golly.sourceforge.io/"),
        ("Wikipedia — Conway's Game of Life", "https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"),
    ]
    body = "".join(
        f'<a href="{u}" target="_blank" rel="noopener" '
        f'style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:14px 0;border-bottom:1px solid {HAIRLINE};text-decoration:none;'
        f'color:{INK};font-size:15.5px;">'
        f'<span>{t}</span><span style="color:{ACTION};">↗</span></a>'
        for t, u in links
    )
    _section_title("LEARN MORE", "延伸阅读")
    ui.html(
        f'<div style="background:#fff;border:1px solid {HAIRLINE};border-radius:16px;'
        f'padding:6px 24px;">{body}</div>'
    )
    ui.html(
        f'<div style="text-align:center;color:{MUTED};font-size:13px;padding:20px 0;">'
        f'本页所有动画均由内置生命引擎按 B3/S23 实时演算并编码为 SVG · '
        f'Built with NiceGUI × Viser × spconv</div>'
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def __glider_seed():
    # 放在左上角的滑翔机，朝右下方滑行
    return [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def _anim_centered(offsets, frames, dt, pad=2, wrap=True, cell=14,
                   rows_min=0, cols_min=0):
    """把以中心为原点的偏移搬到网格并生成动画。"""
    cells, rows, cols = _normalize(offsets, pad=pad)
    rows = max(rows, rows_min)
    cols = max(cols, cols_min)
    return animated_svg(cells, rows=rows, cols=cols, frames=frames, wrap=wrap,
                        cell=cell, dt=dt)


def _still_strip(names):
    """把若干静物并排画成一条静态展示带。"""
    svgs = "".join(
        f'<div style="line-height:0;">{static_svg(PATTERN_OFFSETS[n], cell=13)}</div>'
        for n in names
    )
    return (
        f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">'
        f"{svgs}</div>"
    )


def _escape(text):
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
