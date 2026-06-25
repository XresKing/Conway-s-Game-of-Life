import numpy as np

from gamelogic import GameLogic
from presets import PATTERN_NAMES, pattern_coords


def alive_cells(frame):
    """返回当前帧活细胞坐标 (b, y, x) """
    idx = frame.indices.detach().cpu().numpy()
    return sorted(map(tuple, idx[:, 1:].tolist()))


def run_pattern(game_logic, pattern="glider", bounds=(100, 100), loops=5, center=None):
    H, W = bounds
    if center is None:
        center = (H // 2, W // 2)

    coords = pattern_coords(pattern, center=center)
    features = np.ones((coords.shape[0], 1), dtype=np.float32)

    print(f"\n========== 预设: {pattern} | 边界: {H}x{W} | 中心: {center} ==========")
    frame, conv = game_logic.sparse_2D(
        cell_coordinates=coords,
        cell_features=features,
        spatial_shape=[H, W],
        batch_size=1,
        channels=1,
    )

    print("第 0 帧活着的细胞 (y,x):", alive_cells(frame))
    for step in range(1, loops + 1):
        frame = game_logic.update_frame(frame, conv)
        cells = alive_cells(frame)
        print(f"第 {step} 帧活着的细胞 (y,x):", cells)
        if len(cells) == 0:
            print("种群已灭绝，提前停止。")
            break


def main(loops=5, bounds=(100, 100)):
    """遍历所有预设，逐个演示调用。"""
    game_logic = GameLogic()
    for name in PATTERN_NAMES:
        run_pattern(game_logic, pattern=name, bounds=bounds, loops=loops)


if __name__ == "__main__":
    main()
