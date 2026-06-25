import math

import numpy as np
import trimesh
import viser
from nicegui import ui

from gamelogic import GameLogic, alive_coords
from presets import (
    PATTERN_NAMES,
    PATTERN_NAMES_3D,
    pattern_coords,
    pattern_coords_3d,
)


# ----------------------------------------------------------------------------
# Viser 渲染端
# ----------------------------------------------------------------------------
class ViserRenderer:
    CELL_COLOR = (41, 151, 255)  # #2997ff —— DESIGN.md 的 Sky Link Blue，黑底上呈发光感

    def __init__(self, host="0.0.0.0", port=9000):
        self.server = viser.ViserServer(host=host, port=port)
        # 隐藏 logo / 分享按钮，并且全程不添加任何 Viser GUI 元素，
        # 使右上角开发者小窗最小化（1.0.30 无法彻底删除该容器）。
        self.server.gui.configure_theme(
            show_logo=False,
            show_share_button=False,
            dark_mode=True,
            control_layout="collapsible",
        )

        box = trimesh.creation.box(extents=(0.9, 0.9, 0.9))  # 0.9 留缝露出网格线
        self._verts = box.vertices.astype(np.float32)
        self._faces = box.faces.astype(np.uint32)

        self._cells = None          # 当前活细胞批量网格句柄
        self._paint_cb = None       # 画笔回调 (y, x) -> None
        self._locking = False       # 相机摆放重入保护
        self.mode = "2D"            # "2D" | "3D"
        self.H = 50
        self.W = 50
        self.D = 30                 # 仅 3D 使用的深度

        # 世界 up 设为 -z：2D 棋盘平铺在 z=0 平面、点击拾取以 z=0 求交。
        self.server.scene.set_up_direction("-z")

        # 高环境光 + 柔和方向光：让方块以接近自发光的高亮平铺呈现
        self.server.scene.add_light_ambient("/light/ambient", intensity=1.6)
        self.server.scene.add_light_directional(
            "/light/dir", intensity=0.8, position=(0.0, 0.0, 100.0)
        )

        self.server.on_client_connect(self._on_connect)
        self.server.scene.on_click()(self._on_click)
        self._build_scene()

    # --- 画笔 ---------------------------------------------------------------
    def set_paint_callback(self, cb):
        self._paint_cb = cb

    def _on_click(self, event):
        # 仅 2D 支持平面点击绘制；3D 平面拾取无意义，忽略
        if self._paint_cb is None or self.mode != "2D":
            return
        ro, rd = event.ray_origin, event.ray_direction
        if ro is None or rd is None or abs(rd[2]) < 1e-9:
            return
        t = -ro[2] / rd[2]          # 与 z=0 平面求交
        if t <= 0:
            return
        wx = ro[0] + t * rd[0]
        wy = ro[1] + t * rd[1]
        # 棋盘以原点为中心 → 世界坐标反算回格子索引
        x = int(math.floor(wx + self.W / 2.0))
        y = int(math.floor(wy + self.H / 2.0))
        self._paint_cb(y, x)

    # --- 场景（网格 / 包围盒）-----------------------------------------------
    def _build_scene(self):
        self.server.scene.remove_by_name("/grid")
        self.server.scene.remove_by_name("/bbox")
        if self.mode == "2D":
            # 棋盘以世界原点为中心：客户端默认 look_at 即原点，避免初始视角偏到盘外（黑屏）
            self.server.scene.add_grid(
                "/grid",
                width=float(self.W),
                height=float(self.H),
                plane="xy",
                cell_size=1.0,
                cell_color=(90, 90, 100),
                section_size=10.0,
                section_color=(140, 140, 155),
                plane_color=(0, 0, 0),
                plane_opacity=1.0,
                position=(0.0, 0.0, 0.0),
            )
        else:
            # 3D：底面网格 + 线框包围盒，便于在空间中定位
            self.server.scene.add_grid(
                "/grid",
                width=float(self.W),
                height=float(self.H),
                plane="xy",
                cell_size=1.0,
                cell_color=(70, 70, 82),
                section_size=10.0,
                section_color=(110, 110, 128),
                plane_color=(0, 0, 0),
                plane_opacity=0.6,
                position=(0.0, 0.0, -self.D / 2.0),
            )
            self._build_bbox()

    def _build_bbox(self):
        # 以原点为中心、尺寸 (W,H,D) 的线框立方体（12 条棱）
        hx, hy, hz = self.W / 2.0, self.H / 2.0, self.D / 2.0
        c = [
            (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
            (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        segs = np.array([[c[a], c[b]] for a, b in edges], dtype=np.float32)
        self.server.scene.add_line_segments(
            "/bbox", points=segs, colors=(90, 90, 110), line_width=1.5
        )

    def set_mode(self, mode):
        mode = "3D" if str(mode).upper() == "3D" else "2D"
        if mode == self.mode:
            return
        self.mode = mode
        if self._cells is not None:
            self._cells.remove()
            self._cells = None
        self._build_scene()
        for client in self.server.get_clients().values():
            self._place_camera(client.camera)

    def set_bounds(self, H, W, D=None):
        H, W = int(H), int(W)
        D = self.D if D is None else int(D)
        if (H, W, D) == (self.H, self.W, self.D):
            return  # 尺寸未变：不重建场景、不动相机（避免每次载入/开始都跳一下）
        self.H, self.W, self.D = H, W, D
        self._build_scene()
        for client in self.server.get_clients().values():
            self._place_camera(client.camera)

    # --- 相机：静态摆放在一定距离外看向原点（不再有任何锁定回调）-----------
    def _camera_distance(self, fov=0.75):
        # 让整盘完整可见，外加 15% 余量
        span = max(self.H, self.W, self.D if self.mode == "3D" else 0)
        return span / (2.0 * math.tan(fov / 2.0)) * 1.15 + 5.0

    def _place_camera(self, cam):
        """初始视角：相机置于一定距离外看向原点。2D 为正上方俯视，3D 为斜对角。

        只在连接/切模式/改尺寸时摆放一次，之后用户可自由轨道旋转，
        不再注册 on_update 回拍回调（那是此前相机乱飘的根因）。
        """
        fov = getattr(cam, "fov", 0.75) or 0.75
        dist = self._camera_distance(fov)
        self._locking = True
        try:
            if self.mode == "2D":
                cam.up_direction = (0.0, 1.0, 0.0)
                cam.position = (0.0, 0.0, dist)
            else:
                # 斜对角俯视：相机沿 (1,1,1) 方向退到 dist 处，up 取 +z
                k = dist / math.sqrt(3.0)
                cam.up_direction = (0.0, 0.0, 1.0)
                cam.position = (k, -k, k)
            cam.look_at = (0.0, 0.0, 0.0)
        finally:
            self._locking = False

    def _on_connect(self, client):
        self._place_camera(client.camera)

    # --- 渲染活细胞 ---------------------------------------------------------
    def render(self, coords):
        if self._cells is not None:
            self._cells.remove()
            self._cells = None
        if coords is None or len(coords) == 0:
            return
        coords = np.asarray(coords)
        n = coords.shape[0]
        positions = np.zeros((n, 3), dtype=np.float32)
        if self.mode == "2D":
            # 棋盘以原点为中心：格子索引 (y, x) → 世界坐标
            positions[:, 0] = coords[:, 1].astype(np.float32) - self.W / 2.0 + 0.5  # x
            positions[:, 1] = coords[:, 0].astype(np.float32) - self.H / 2.0 + 0.5  # y
            positions[:, 2] = 0.45                                                  # 坐落在黑底之上
        else:
            # 立方体以原点为中心：体素索引 (z, y, x) → 世界坐标
            positions[:, 0] = coords[:, 2].astype(np.float32) - self.W / 2.0 + 0.5  # x
            positions[:, 1] = coords[:, 1].astype(np.float32) - self.H / 2.0 + 0.5  # y
            positions[:, 2] = coords[:, 0].astype(np.float32) - self.D / 2.0 + 0.5  # z
        wxyzs = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
        self._cells = self.server.scene.add_batched_meshes_simple(
            "/cells",
            vertices=self._verts,
            faces=self._faces,
            batched_wxyzs=wxyzs,
            batched_positions=positions,
            batched_colors=self.CELL_COLOR,
            material="standard",
            flat_shading=True,
        )


# ----------------------------------------------------------------------------
# 仿真协调（编辑态 / 运行态共享状态）
# ----------------------------------------------------------------------------
class GameController:
    def __init__(self, renderer: ViserRenderer):
        self.renderer = renderer
        self.logic = GameLogic()
        self.mode = "2D"             # "2D" | "3D"
        self.H = renderer.H
        self.W = renderer.W
        self.D = renderer.D
        self.paint = set()           # 编辑态初值：2D 为 {(y,x)}，3D 为 {(z,y,x)}
        self.frame = None
        self.conv = None
        self.generation = 0
        self.running = False
        self.max_loops = 100
        self.born = None             # None → 由 gamelogic 按维度取默认规则
        self.survive = None
        renderer.set_paint_callback(self.toggle_cell)
        renderer.set_bounds(self.H, self.W, self.D)

    @property
    def ndim(self):
        return 3 if self.mode == "3D" else 2

    def _in_bounds(self, cell):
        if self.mode == "2D":
            y, x = cell
            return 0 <= y < self.H and 0 <= x < self.W
        z, y, x = cell
        return 0 <= z < self.D and 0 <= y < self.H and 0 <= x < self.W

    def _coords_array(self):
        if not self.paint:
            return np.zeros((0, self.ndim), dtype=np.int32)
        return np.array(sorted(self.paint), dtype=np.int32)

    def _render_edit(self):
        self.renderer.render(self._coords_array())

    def set_mode(self, mode):
        if self.running:
            return
        mode = "3D" if str(mode).upper() == "3D" else "2D"
        if mode == self.mode:
            return
        self.mode = mode
        self.frame = None
        self.conv = None
        self.generation = 0
        self.paint = set()
        self.renderer.set_mode(mode)
        self._render_edit()

    def set_rules(self, born, survive):
        # 传 None 表示恢复该维度默认规则
        self.born = None if born is None else frozenset(int(b) for b in born)
        self.survive = None if survive is None else frozenset(int(s) for s in survive)

    def toggle_cell(self, y, x):
        # 仅 2D 画笔使用
        if self.running or self.mode != "2D" or not self._in_bounds((y, x)):
            return
        key = (y, x)
        self.paint.discard(key) if key in self.paint else self.paint.add(key)
        self._render_edit()

    def load_preset(self, name):
        if self.running:
            return
        if self.mode == "2D":
            coords = pattern_coords(name, center=(self.H // 2, self.W // 2))
        else:
            coords = pattern_coords_3d(
                name, center=(self.D // 2, self.H // 2, self.W // 2)
            )
        self.paint = {
            tuple(int(v) for v in cell) for cell in coords if self._in_bounds(cell)
        }
        self._render_edit()

    def random_fill(self, density=0.15):
        if self.running:
            return
        rng = np.random.default_rng()
        if self.mode == "2D":
            mask = rng.random((self.H, self.W)) < density
            zs = np.argwhere(mask)
            self.paint = {(int(y), int(x)) for y, x in zs}
        else:
            mask = rng.random((self.D, self.H, self.W)) < density
            zs = np.argwhere(mask)
            self.paint = {(int(z), int(y), int(x)) for z, y, x in zs}
        self._render_edit()

    def clear(self):
        if self.running:
            return
        self.paint = set()
        self._render_edit()

    def set_bounds(self, H, W, D=None):
        if self.running:
            return
        self.H, self.W = int(H), int(W)
        if D is not None:
            self.D = int(D)
        self.paint = {cell for cell in self.paint if self._in_bounds(cell)}
        self.renderer.set_bounds(self.H, self.W, self.D)
        self._render_edit()

    def start(self):
        if self.running:
            return True
        coords = self._coords_array()
        if len(coords) == 0:
            return False
        feats = np.ones((len(coords), 1), dtype=np.float32)
        if self.mode == "2D":
            self.frame, self.conv = self.logic.sparse_2D(
                cell_coordinates=coords,
                cell_features=feats,
                spatial_shape=[self.H, self.W],
                batch_size=1,
                channels=1,
            )
        else:
            self.frame, self.conv = self.logic.sparse_3D(
                cell_coordinates=coords,
                cell_features=feats,
                spatial_shape=[self.D, self.H, self.W],
                batch_size=1,
                channels=1,
            )
        self.generation = 0
        self.running = True
        return True

    def step(self):
        if not self.running or self.frame is None:
            return
        self.frame = self.logic.update_frame(
            self.frame, self.conv, born=self.born, survive=self.survive
        )
        coords = alive_coords(self.frame)
        self.generation += 1
        self.renderer.render(coords)
        if len(coords) == 0 or self.generation >= self.max_loops:
            self.running = False

    def pause(self):
        self.running = False

    def reset(self):
        self.running = False
        self.frame = None
        self.conv = None
        self.generation = 0
        self._render_edit()

    def alive_count(self):
        # 仿真已建立帧（运行中或刚停止）时以帧为准；否则显示编辑态初值数量
        if self.frame is not None:
            return int(self.frame.indices.shape[0])
        return len(self.paint)


# ----------------------------------------------------------------------------
# 启动 Viser + 构建 NiceGUI 界面
# ----------------------------------------------------------------------------
renderer = ViserRenderer(port=9000)
controller = GameController(renderer)

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  html, body { margin: 0; height: 100%; background: #f5f5f7;
    font-family: 'Inter', system-ui, -apple-system, sans-serif; color: #1d1d1f; }
  .nicegui-content { padding: 0 !important; height: 100vh; width: 100vw;
    max-width: 100vw; overflow: hidden; }
  .gol-title { font-size: 21px; font-weight: 600; letter-spacing: -0.2px; }
  .gol-sub { font-size: 12px; color: #7a7a7a; }
  .gol-label { font-size: 14px; font-weight: 600; color: #1d1d1f; }
  .gol-frame { width: 100%; height: 100vh; border: 0; display: block; }
</style>
"""


@ui.page("/")
def index():
    ui.add_head_html(HEAD)
    ui.colors(primary="#0066cc")  # DESIGN.md 唯一强调色 Action Blue

    with ui.row().classes("no-wrap gap-0").style(
        "width:100vw; height:100vh; flex-wrap:nowrap;"
    ):
        # ---- 左栏 ----------------------------------------------------------
        with ui.column().classes("h-full gap-4").style(
            "width:320px; min-width:320px; background:#ffffff; "
            "padding:24px; overflow-y:auto; border-right:1px solid #e0e0e0;"
        ):
            with ui.column().classes("gap-0"):
                ui.label("康威生命游戏").classes("gol-title")
                ui.label("spconv · NiceGUI × Viser").classes("gol-sub")

            ui.label("维度模式").classes("gol-label")
            mode_toggle = ui.toggle(
                {"2D": "2D 平面", "3D": "3D 立体"}, value="2D"
            ).props("no-caps").classes("w-full")

            preset = ui.select(
                PATTERN_NAMES, value="glider", label="预设图案"
            ).props("outlined dense").classes("w-full")

            ui.label("边界").classes("gol-label")
            with ui.row().classes("w-full no-wrap gap-3"):
                h_in = ui.number("高度 H", value=50, min=10, max=200, step=1).props(
                    "outlined dense"
                ).classes("w-full")
                w_in = ui.number("宽度 W", value=50, min=10, max=200, step=1).props(
                    "outlined dense"
                ).classes("w-full")
            d_in = ui.number("深度 D（仅 3D）", value=30, min=10, max=120, step=1).props(
                "outlined dense"
            ).classes("w-full")
            d_in.set_visibility(False)

            with ui.row().classes("w-full no-wrap gap-3"):
                loops_in = ui.number(
                    "循环次数", value=100, min=1, max=10000, step=1
                ).props("outlined dense").classes("w-full")
                density_in = ui.number(
                    "随机密度", value=0.15, min=0.01, max=0.9, step=0.01
                ).props("outlined dense").classes("w-full")

            ui.label("规则（不含自身的邻居数；留空用默认）").classes("gol-label")
            with ui.row().classes("w-full no-wrap gap-3"):
                born_in = ui.input("出生 Born", value="3").props(
                    "outlined dense"
                ).classes("w-full")
                survive_in = ui.input("存活 Survive", value="2,3").props(
                    "outlined dense"
                ).classes("w-full")

            ui.label("速度（每帧毫秒）").classes("gol-label")
            speed = ui.slider(min=50, max=1000, value=150, step=10).props("label-always")

            # 操作按钮
            with ui.column().classes("w-full gap-2").style("margin-top:8px;"):
                ui.button("载入预设", on_click=lambda: _load()).props(
                    "outline rounded no-caps"
                ).classes("w-full")
                with ui.row().classes("w-full no-wrap gap-2"):
                    ui.button("随机填充", on_click=lambda: _random()).props(
                        "outline rounded no-caps"
                    ).classes("w-full")
                    ui.button("清空", on_click=lambda: _clear()).props(
                        "outline rounded no-caps"
                    ).classes("w-full")
                ui.button("▶ 开始循环", on_click=lambda: _start()).props(
                    "unelevated rounded no-caps"
                ).classes("w-full")
                with ui.row().classes("w-full no-wrap gap-2"):
                    ui.button("暂停", on_click=lambda: _pause()).props(
                        "outline rounded no-caps"
                    ).classes("w-full")
                    ui.button("单步", on_click=lambda: _single()).props(
                        "outline rounded no-caps"
                    ).classes("w-full")
                ui.button("重置", on_click=lambda: _reset()).props(
                    "flat rounded no-caps"
                ).classes("w-full")

            status = ui.label().classes("gol-sub").style("margin-top:8px;")
            tip = ui.label(
                "提示：2D 可在右侧画布点击格子绘制初态；3D 用预设或随机填充生成初态。"
            ).classes("gol-sub")

        # ---- 右屏：Viser iframe -------------------------------------------
        # 用 ui.element('iframe') 直接生成真实 <iframe> DOM。
        # 不能用 ui.html()：其默认 sanitize=True 会经 DOMPurify 把 <iframe> 整个剥离（→黑屏）。
        with ui.element("div").style(
            "flex:1 1 0; height:100vh; min-width:0; background:#000;"
        ):
            ui.element("iframe").props(
                'src="http://localhost:9000" allow="fullscreen"'
            ).style("width:100%; height:100vh; border:0; display:block;")

    # ---- 交互逻辑 ----------------------------------------------------------
    loop_timer = {"t": None}

    # 默认规则文本，用于切模式时回填
    DEFAULTS = {
        "2D": {"born": "3", "survive": "2,3"},
        "3D": {"born": "6", "survive": "5,6,7"},
    }

    def _parse_rule(text):
        # "2,3" → frozenset{2,3}；空串 → None（用默认）
        s = (text or "").strip()
        if not s:
            return None
        out = set()
        for part in s.replace("，", ",").split(","):
            part = part.strip()
            if part:
                out.add(int(part))
        return out or None

    def _apply_rules():
        try:
            controller.set_rules(_parse_rule(born_in.value), _parse_rule(survive_in.value))
        except ValueError:
            ui.notify("规则需为逗号分隔的整数，例如 2,3", type="warning")

    def _apply_bounds():
        controller.set_bounds(int(h_in.value), int(w_in.value), int(d_in.value))

    def _stop_timer():
        if loop_timer["t"] is not None:
            loop_timer["t"].active = False

    def _on_mode(e):
        _stop_timer()
        is_3d = e.value == "3D"
        d_in.set_visibility(is_3d)
        preset.options = PATTERN_NAMES_3D if is_3d else PATTERN_NAMES
        preset.value = (PATTERN_NAMES_3D if is_3d else PATTERN_NAMES)[0]
        preset.update()
        born_in.value = DEFAULTS[e.value]["born"]
        survive_in.value = DEFAULTS[e.value]["survive"]
        controller.set_mode(e.value)

    mode_toggle.on_value_change(_on_mode)

    def _load():
        _stop_timer()
        controller.reset()
        _apply_bounds()
        controller.load_preset(preset.value)

    def _random():
        _stop_timer()
        controller.reset()
        _apply_bounds()
        controller.random_fill(float(density_in.value))

    def _clear():
        _stop_timer()
        controller.reset()
        _apply_bounds()
        controller.clear()

    def _start():
        _apply_bounds()
        _apply_rules()
        controller.max_loops = int(loops_in.value)
        if not controller.start():
            ui.notify("初态为空：先载入预设、随机填充或在画布上绘制。", type="warning")
            return
        interval = max(0.05, float(speed.value) / 1000.0)
        if loop_timer["t"] is not None:
            loop_timer["t"].cancel()
        loop_timer["t"] = ui.timer(interval, _tick, active=True)

    def _tick():
        if not controller.running:
            _stop_timer()
            return
        controller.step()

    def _pause():
        _stop_timer()
        controller.pause()

    def _single():
        controller.max_loops = int(loops_in.value)
        if not controller.running:
            _apply_bounds()
            _apply_rules()
            if not controller.start():
                ui.notify("初态为空：先载入预设、随机填充或在画布上绘制。", type="warning")
                return
        else:
            controller.step()

    def _reset():
        _stop_timer()
        controller.reset()

    def _refresh_status():
        mode = "运行中" if controller.running else "编辑中"
        bound = (
            f"{controller.H}×{controller.W}"
            if controller.mode == "2D"
            else f"{controller.D}×{controller.H}×{controller.W}"
        )
        status.text = (
            f"模式 {controller.mode} · 状态：{mode} · 代数 {controller.generation} · "
            f"存活 {controller.alive_count()} · 边界 {bound}"
        )

    ui.timer(0.2, _refresh_status)  # 轮询刷新（画笔点击发生在 Viser 线程）


if __name__ == "__main__":
    ui.run(host="0.0.0.0", port=8080, title="康威生命游戏", reload=False, show=True)
