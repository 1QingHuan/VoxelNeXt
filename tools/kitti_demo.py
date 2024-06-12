# kitti:
# python kitti_demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ./pv_rcnn_8369.pth --data_path ../data/kitti/training/velodyne
# waymo:
# python kitti_demo.py --cfg_file cfgs/waymo_models/pv_rcnn.yaml --ckpt ./pv_rcnn_8369.pth --data_path ../data/waymo/waymo_processed_data_v0_5_0/segment-1005081002024129653_5313_150_5333_150_with_camera_labels --ext '.npy'


import argparse
import glob
from pathlib import Path
import time

import open3d as o3d
import torch
import matplotlib
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# 数据集类：处理点云数据 implements DatasetTemplate
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    # overide __getitem__ method
    # 从文件中读取点云数据，返回数据字典
    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            print(f"self.src_feature_list: {self.point_feature_encoder.src_feature_list}")
            print(f"points shape: {points.shape}")
            # kitti: 'x', 'y', 'z', 'intensity'
            points = points.reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            # print(f"self.src_feature_list: {self.point_feature_encoder.src_feature_list}")
            # print(f"points shape[-1]: {points.shape[-1]}")
            # waymo: 'x', 'y', 'z', 'intensity', 'elongation'
            points = points[:, :5]
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # --cfg_file：指定配置文件
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    # --data_path：指定点云数据文件或目录
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    # --ckpt：指定预训练模型
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    # --ext：指定点云数据文件的扩展名
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


# 检测框绘制颜色
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


# 获取坐标颜色 -label
def get_coor_colors(obj_labels):
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


# 绘制场景
def draw_scenes(vis, points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # 清除上一帧
    vis.clear_geometries()

    # draw origin
    # 绘制坐标系
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = o3d.utility.Vector3dVector(point_colors)
    vis.add_geometry(pts)

    # 绘制gt_boxes：蓝色 行人
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    # 绘制ref_boxes：绿色 大型车辆 其余类别的检测框
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.update_renderer()


def translate_boxes_to_open3d_instance(gt_boxes):
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            # 绘制其余类别的检测框
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

    return vis


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # 创建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # 加载预训练模型
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # 关闭梯度计算
    with torch.no_grad():
        # 对每个样本进行预测
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            # 将数据字典转换为批量数据
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            # 前向传播
            pred_dicts, _ = model.forward(data_dict)

            # 绘制场景
            draw_scenes(
                vis, points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            vis.poll_events()
            vis.update_renderer()

            # time.sleep(0.25)


    vis.destroy_window()
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
