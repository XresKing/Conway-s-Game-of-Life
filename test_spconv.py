import numpy as np
import torch
import spconv.core as sc
import spconv.pytorch as spconv

def test_sparse_3D():
    print("==找 CUDA 环境==")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"检测设备时发生错误: {e}")
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"Pytorch 版本: {torch.__version__}")
    print(f"CUDA 是否可以用： {torch.cuda.is_available()}")

    if device.type != "cuda":
        print("CUDA 不可用")
    
    print("\n==3D 环境测试==")

    '''
    10X10X10的空间，cell_size为1，坐标为整数(Z,Y,X)
    '''
    cell_coordinates = np.array(
        [
            [1,2,2],
            [2,2,2],
            [3,2,2]
        ]
        ,dtype=np.int32
    )
    cell_features = np.array(
        [
            [1.0],
            [1.0],
            [1.0]
        ]
        ,dtype=np.float32
    )
    batch_column = np.zeros((cell_coordinates.shape[0], 1), dtype=np.int32)
    sp_indices = np.hstack([batch_column, cell_coordinates])

    indices_tensor = torch.from_numpy(sp_indices).to(device)
    features_tensor = torch.from_numpy(cell_features).to(device)

    spatial_shape = [10, 10, 10]
    batch_size = 1

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
        return
    
    sub_conv = spconv.SubMConv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False
    ).to(device)

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

def test_sparse_2D():
    print("==找 CUDA 环境==")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"检测设备时发生错误: {e}")
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"Pytorch 版本: {torch.__version__}")
    print(f"CUDA 是否可以用： {torch.cuda.is_available()}")

    if device.type != "cuda":
        print("CUDA 不可用")
    
    print("\n==2D 环境测试==")

    cell_coordinates = np.array(
        [
            [2,2],
            [2,3],
            [3,2]
        ]
        ,dtype=np.int32
    )
    cell_features = np.array(
        [
            [1.0],
            [1.0],
            [1.0]
        ]
        ,dtype=np.float32
    )
    batch_column = np.zeros((cell_coordinates.shape[0], 1), dtype=np.int32)
    sp_indices = np.hstack([batch_column, cell_coordinates])

    indices_tensor = torch.from_numpy(sp_indices).to(device)
    features_tensor = torch.from_numpy(cell_features).to(device)
    
    spatial_shape = [10,10]
    batch_size = 1

    try:
        input_tensor = spconv.SparseConvTensor(
            features = features_tensor,
            indices = indices_tensor,
            spatial_shape = spatial_shape,
            batch_size = batch_size
        )
    except Exception as e:
        print(f"创建 SparseConvTensor 时发生错误: {e}")
        return
    
    sub_conv = spconv.SubMConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False
    ).to(device)

    with torch.no_grad():
        sub_conv.weight.fill_(1.0)

    print("\n==正在 GPU 中执行子流形系数卷积==")
    output_tensor = sub_conv(input_tensor)
    out_features = output_tensor.features.detach().cpu().numpy()
    out_indices = output_tensor.indices.detach().cpu().numpy()

    print(f"\n输出特征形状: {out_features.shape}")
    for i in range(len(out_indices)):
        b,y,x = out_indices[i]
        neighbor_count = out_features[i][0]
        print(f"活跃细胞位置: (Batch: {b}, Y: {y}, X: {x}), 邻居数量(含自身): {neighbor_count}")

if __name__ == "__main__":
    test_sparse_2D()