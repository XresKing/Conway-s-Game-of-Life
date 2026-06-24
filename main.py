import numpy as np
from gamelogic import GameLogic

def alive_cells(frame):
    idx = frame.indices.detach().cpu().numpy()
    return sorted(map(tuple, idx[:, 1:].tolist()))

def make_pattern(relative_coords, center=(50, 50), dtype = np.int32):
    return np.array([[center[0] + dy, center[1] + dx] for dy, dx in relative_coords], dtype=dtype)

PATTERNS = {
    "block": make_pattern([(0,0), (0,1), (1,0), (1,1)]),
    
    "blinker": make_pattern([(0,0), (0,1), (0,2)]),
    
    "beehive": make_pattern([
        (0,1), (0,2),
        (1,0), (1,3),
        (2,1), (2,2)
    ]),
    
    "toad": make_pattern([
        (0,1), (0,2), (0,3),
        (1,0), (1,1), (1,2)
    ]),
    
    "beacon": make_pattern([
        (0,0), (0,1),
        (1,0), (1,1),
        (2,2), (2,3),
        (3,2), (3,3)
    ]),
    
    "glider": make_pattern([
        (0,1),
        (1,2),
        (2,0), (2,1), (2,2)
    ]),
    
    "lwss": make_pattern([  # 轻量级飞船
        (-2, 0),
        (-1,-2), (-1, 1),
        ( 1,-2), ( 1, 1),
        ( 2,-1), ( 2, 0), ( 2, 1)
    ]),
}


def main(loops = 3):
    game_logic = GameLogic()

    frame, conv = game_logic.sparse_2D(
            cell_coordinates=np.array([[2, 2], [2, 3], [3, 2], [9, 9], [9, 10], [10, 9]], dtype=np.int32),
            cell_features=np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
            spatial_shape=[100, 100],
            batch_size=1,
            channels=1
    )

    # 第 0 帧 
    print(f'第 0 帧活着的细胞 (b,y,x):', alive_cells(frame))

    for step in range(1, loops + 1):
        frame = game_logic.update_frame(frame, conv)
        cells = alive_cells(frame)
        print(f'第 {step} 帧活着的细胞 (b,y,x):', cells)
        if len(cells) == 0:
            print('种群已灭绝，提前停止。')
            break

if __name__ == "__main__":
    main()