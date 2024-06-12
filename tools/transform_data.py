import numpy as np
import os


def transform_point_cloud(points):
    """
    转换点云数据的坐标系并处理强度信息。

    Args:
        points (np.ndarray): 原始点云数据，形状为 (num_points, 6)

    Returns:
        np.ndarray: 处理后的点云数据，形状为 (num_points, 4)
    """
    # 提取前四个维度: x, y, z 和 intensity（如果有）
    if points.shape[1] > 4:
        points = points[:, :4]

    # 如果没有强度信息，添加一个强度列并设置为零
    if points.shape[1] == 3:
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))

    # 确保强度信息存在，并在范围 [0, 1] 内
    points[:, 3] = 0  # 如果没有强度信息，将其设置为零

    return points


def process_npy_file(input_path):
    """
    读取 .npy 文件，处理点云数据，并保存为另一个 .npy 文件。

    Args:
        input_path (str): 输入 .npy 文件的路径
    """
    # 读取原始点云数据
    points = np.load(input_path)

    # 打印实际数据形状以便调试
    print(f"Actual data shape: {points.shape}")

    # 确保点云数据为 (num_points, 3) 或 (num_points, 4) 或更多维度
    if points.shape[1] < 3:
        raise ValueError(f"Invalid point cloud data shape. Expected at least (num_points, 3), but got {points.shape}")

    # 处理点云数据
    transformed_points = transform_point_cloud(points)

    # 生成输出文件路径
    base_name = os.path.basename(input_path)
    dir_name = os.path.dirname(input_path)
    output_path = os.path.join(dir_name, f"transformed_{base_name}")

    # 保存处理后的点云数据
    np.save(output_path, transformed_points)
    print(f"Transformed point cloud saved to: {output_path}")


# 示例使用
input_path = "../data/waymo/waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy"
process_npy_file(input_path)
