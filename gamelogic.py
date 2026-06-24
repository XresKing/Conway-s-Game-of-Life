import numpy as np
import torch
import spconv.core as sc
import spconv.pytorch as spconv

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
            valid_mask = (cell_coordinates[:, 0] > 0) & (cell_coordinates[:, 0] < H) & (cell_coordinates[:, 1] > 0) & (cell_coordinates[:, 1] < W)
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

    def sparse_3D(self,cell_coordinates, cell_features, spatial_shape, conv_type = 'Sparse',batch_size = 1,channels = 1):
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
        
        if conv_type == 'Sparse':
            sub_conv = spconv.SparseConv3d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                bias=False
            ).to(self.device)
        else:
            sub_conv = spconv.SubMConv3d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                bias=False
            ).to(self.device)

        with torch.no_grad():
            sub_conv.weight.fill_(1.0)

        print("\n==正在 GPU 中执行子流形系数卷积==")
        output_tensor = sub_conv(input_tensor)
        out_features = output_tensor.features.detach().cpu().numpy()
        out_indices = output_tensor.indices.detach().cpu().numpy()
        
        print(f"\n输出特征形状: {out_features.shape}")
        for i in range(len(out_indices)):
            b,z,y,x = out_indices[i]
            neighbor_count = out_features[i][0]
            print(f"活跃细胞位置: (Batch: {b}, Z: {z}, Y: {y}, X: {x}), 邻居数量(含自身): {neighbor_count}")

    def update_frame(self, input_tensor, sparse_conv_layer):
        output_tensor = sparse_conv_layer(input_tensor)
        candidate_indices = output_tensor.indices.detach().cpu().numpy()
        neighbor_counts = output_tensor.features[:,0].detach().cpu().numpy()

        H, W = input_tensor.spatial_shape
        cand_y = candidate_indices[:, 1]
        cand_x = candidate_indices[:, 2]
        valid_mask = (cand_y >= 0) & (cand_y <= H) & (cand_x >= 0) & (cand_x <= W)
        candidate_indices = candidate_indices[valid_mask]
        neighbor_counts = neighbor_counts[valid_mask]

        def _flatten(idx):
            b = idx[:, 0].astype(np.int64)
            y = idx[:, 1].astype(np.int64)
            x = idx[:, 2].astype(np.int64)
            return (b * H + y) * W + x

        input_indices = input_tensor.indices.detach().cpu().numpy()
        occupancy = np.zeros(input_tensor.batch_size * H * W, dtype=bool)
        occupancy[_flatten(input_indices)] = True
        was_alive_mask = occupancy[_flatten(candidate_indices)]

        was_died_mask = ~was_alive_mask

        #此处包含了自身，因此是3或者4
        keep_alive = was_alive_mask & ((neighbor_counts == 3) | (neighbor_counts == 4))
        #只有当周围刚好2个细胞时才能复活
        reborn = was_died_mask & (neighbor_counts == 3)
        total_alive_mask = reborn | keep_alive

        total_indices = candidate_indices[total_alive_mask]
        total_features = torch.ones((total_indices.shape[0], 1),dtype=torch.float32).to(self.device)
        total_indices = torch.from_numpy(total_indices.astype(np.int32)).to(self.device)

        total_frame_tensor = spconv.SparseConvTensor(
            features=total_features,
            indices=total_indices,
            spatial_shape=input_tensor.spatial_shape,
            batch_size=input_tensor.batch_size
        )

        return total_frame_tensor
