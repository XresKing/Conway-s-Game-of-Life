import numpy as np
import torch
import spconv.core as sc
import spconv.pytorch as spconv

# 默认规则（不含自身的邻居数）：2D 经典 B3/S23；3D 采用 Bays 5766（B6/S567）
DEFAULT_RULES = {
    2: {"born": frozenset({3}), "survive": frozenset({2, 3})},
    3: {"born": frozenset({6}), "survive": frozenset({5, 6, 7})},
}

class GameLogic:
    def __init__(self):
        print("\n==初始化 GameLogic==")
        print("\n==找 CUDA 环境==")

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"检测设备时发生错误: {e}")
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        print(f"Pytorch 版本: {torch.__version__}")
        print(f"CUDA 是否可以用： {torch.cuda.is_available()}")

        if self.device.type != "cuda":
            print("CUDA 不可用")

    def sparse_2D(self, cell_coordinates, cell_features, spatial_shape, input_tensor = None, batch_size = 1,channels = 1):
        if input_tensor is None:
            H, W = spatial_shape
            valid_mask = (cell_coordinates[:, 0] >= 0) & (cell_coordinates[:, 0] < H) & (cell_coordinates[:, 1] >= 0) & (cell_coordinates[:, 1] < W)
            cell_coordinates = cell_coordinates[valid_mask]
            batch_column = np.zeros((cell_coordinates.shape[0], 1), dtype=np.int32)
            sp_indices = np.hstack([batch_column, cell_coordinates])

            indices_tensor = torch.from_numpy(sp_indices).to(self.device)
            features_tensor = torch.from_numpy(cell_features).to(self.device)

            try:
                input_tensor = spconv.SparseConvTensor(
                    features = features_tensor,
                    indices = indices_tensor,
                    spatial_shape = spatial_shape,
                    batch_size = batch_size
                )

                print("==SparseConvTensor 创建成功==")
                print(f"\n当前活跃细胞数量 {input_tensor.features.shape[0]}")
            except Exception as e:
                print(f"创建 SparseConvTensor 时发生错误: {e}")
                return None
            

        sub_conv_sparse = spconv.SparseConv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ).to(self.device)
        sub_conv_subm = spconv.SubMConv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                bias=False
            ).to(self.device)

        with torch.no_grad():
            sub_conv_sparse.weight.fill_(1.0)
            sub_conv_subm.weight.fill_(1.0)

        output_tensor = sub_conv_subm(input_tensor)
        out_features = output_tensor.features.detach().cpu().numpy()
        out_indices = output_tensor.indices.detach().cpu().numpy()

        print(f"\n输出特征形状: {out_features.shape}")
        for i in range(len(out_indices)):
            b,y,x = out_indices[i]
            neighbor_count = out_features[i][0]
            print(f"活跃细胞位置: (Batch: {b}, Y: {y}, X: {x}), 邻居数量(含自身): {neighbor_count}")
        
        return [input_tensor,sub_conv_sparse]

    def sparse_3D(self, cell_coordinates, cell_features, spatial_shape, input_tensor = None, batch_size = 1, channels = 1):
        if input_tensor is None:
            # spatial_shape = [D, H, W]，坐标列序为 (z, y, x)，逐轴裁剪到界内
            D, H, W = spatial_shape
            valid_mask = (
                (cell_coordinates[:, 0] >= 0) & (cell_coordinates[:, 0] < D)
                & (cell_coordinates[:, 1] >= 0) & (cell_coordinates[:, 1] < H)
                & (cell_coordinates[:, 2] >= 0) & (cell_coordinates[:, 2] < W)
            )
            cell_coordinates = cell_coordinates[valid_mask]
            cell_features = cell_features[valid_mask]
            batch_column = np.zeros((cell_coordinates.shape[0], 1), dtype = np.int32)
            sp_indices = np.hstack([batch_column,cell_coordinates])

            indices_tensor = torch.from_numpy(sp_indices).to(self.device)
            features_tensor = torch.from_numpy(cell_features).to(self.device)

            try:
                input_tensor = spconv.SparseConvTensor(
                    features = features_tensor,
                    indices = indices_tensor,
                    spatial_shape = spatial_shape,
                    batch_size = batch_size
                )
                print("== SparseConvTensor 创建成功")
                print(f"\n现在有 {input_tensor.features.shape[0]} 个活细胞")
            except Exception as e:
                print(f"创建 SparseConvTensor 失败 {e}")
                return None

        sub_conv_sparse = spconv.SparseConv3d(
            in_channels=channels,
            out_channels=1,
            kernel_size=3,
            stride = 1,
            padding=1,
            bias=False
        ).to(self.device)
        sub_conv_subm = spconv.SubMConv3d(
            in_channels=channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ).to(self.device)

        with torch.no_grad():
            sub_conv_sparse.weight.fill_(1.0)
            sub_conv_subm.weight.fill_(1.0)

        output_tensor = sub_conv_sparse(input_tensor)
        out_features = output_tensor.features.detach().cpu().numpy()
        out_indices = output_tensor.indices.detach().cpu().numpy()

        print(f"\n输出特征形状: {out_features.shape}")
        for i in range(len(out_indices)):
            b,z,y,x = out_indices[i]
            neighbor_count = out_features[i][0]
            print(f"活跃细胞位置: (Batch: {b},Z: {z} Y: {y}, X: {x}), 邻居数量(含自身): {neighbor_count}")
        
        return [input_tensor,sub_conv_sparse]

    def update_frame(self, input_tensor, sparse_conv_layer, born=None, survive=None):
        """演化一帧，2D/3D 通用。

        born/survive 为"不含自身"的邻居数集合；不传时按维度取 DEFAULT_RULES。
        卷积权重全 1，输出特征为含自身的计数，故先减去自身得到真实邻居数。
        """
        ndim = len(input_tensor.spatial_shape)
        rules = DEFAULT_RULES[ndim]
        born = rules["born"] if born is None else frozenset(born)
        survive = rules["survive"] if survive is None else frozenset(survive)

        output_tensor = sparse_conv_layer(input_tensor)
        candidate_indices = output_tensor.indices.detach().cpu().numpy()
        neighbor_counts = output_tensor.features[:, 0].detach().cpu().numpy()

        shape = np.asarray(input_tensor.spatial_shape, dtype=np.int64)
        spatial = candidate_indices[:, 1:]  # 去掉 batch 列，剩 ndim 个空间轴
        valid_mask = np.all((spatial >= 0) & (spatial < shape), axis=1)
        candidate_indices = candidate_indices[valid_mask]
        neighbor_counts = neighbor_counts[valid_mask]

        # 行优先把 (b, axis0, axis1, ...) 压成一维下标，用于占用查询
        def _flatten(idx):
            flat = idx[:, 0].astype(np.int64)  # batch
            for axis in range(ndim):
                flat = flat * shape[axis] + idx[:, axis + 1].astype(np.int64)
            return flat

        total_cells = int(input_tensor.batch_size * np.prod(shape))
        input_indices = input_tensor.indices.detach().cpu().numpy()
        occupancy = np.zeros(total_cells, dtype=bool)
        occupancy[_flatten(input_indices)] = True
        was_alive_mask = occupancy[_flatten(candidate_indices)]

        # 真实邻居数 = 含自身计数 - 自身（活细胞才含自身）
        neighbors = neighbor_counts - was_alive_mask.astype(neighbor_counts.dtype)

        survive_arr = np.isin(neighbors, list(survive))
        born_arr = np.isin(neighbors, list(born))
        keep_alive = was_alive_mask & survive_arr
        reborn = (~was_alive_mask) & born_arr
        total_alive_mask = reborn | keep_alive

        total_indices = candidate_indices[total_alive_mask]
        total_features = torch.ones((total_indices.shape[0], 1), dtype=torch.float32).to(self.device)
        total_indices = torch.from_numpy(total_indices.astype(np.int32)).to(self.device)

        total_frame_tensor = spconv.SparseConvTensor(
            features=total_features,
            indices=total_indices,
            spatial_shape=input_tensor.spatial_shape,
            batch_size=input_tensor.batch_size
        )

        return total_frame_tensor


def alive_coords(frame):
    """返回当前帧活细胞的空间坐标 (N, ndim) numpy（去掉 batch 列），2D 为 [y,x]、3D 为 [z,y,x]。"""
    idx = frame.indices.detach().cpu().numpy()
    return idx[:, 1:].astype(np.int32)
