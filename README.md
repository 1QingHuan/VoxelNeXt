VoxelNeXt网络在VoxelNet的基础上，引入了MeanVFE体素特征提取模块，对点云数据进行两次下采样，在不明显增加网络计算量的情况下，提高了网络的感受野，增强了网络的特征提取能力。
同时，VoxelNeXt网络将3D稀疏体素特征提取和2D稠密特征提取通过Sparse Max Pooling相结合，充分利用了点云数据的空间信息和特征信息，提高了网络的检测性能。

VoxelNext项目继承自OpenPCDet项目，因此在使用VoxelNext项目之前，需要先安装OpenPCDet项目。
```python
python setup.py develop
```

Kitti数据集是用于自动驾驶和计算机视觉研究的基准数据集，包含来自真实世界驾驶场景的多模态传感器数据，包括彩色图像、深度图和激光雷达点云等。

对于Kitti数据集，将其处理为以下形式：
```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```
使用以下方法处理Kitti数据集：
```python
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
对Kitti数据集进行训练：
```bash
cd tools
bash scripts/dist_train.sh 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml
```
对Kitti数据集进行测试：
```bash
cd tools
bash scripts/dist_test.sh 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth
```
使用Kitti Demo:
```python
python kitti_demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ./pv_rcnn_8369.pth --data_path ../data/kitti/training/velodyne
```

NuScenes数据集是由Motional发布的用于自动驾驶研究的综合性数据集，提供了全面的360度感知信息，包括来自6个摄像头的图像、5个雷达和1个激光雷达的点云，以及高精度的GPS/IMU数据，广泛用于3D目标检测、跟踪、场景分割和预测等任务。

本项目采用NuScenes数据集的v1.0-mini版本，该版本包含了NuScenes数据集的一个子集，包括1000个场景、7000个样本、40000个激光雷达扫描和14000个图像，适合用于快速验证算法的有效性。

对于NuScenes数据集，将其处理为以下形式：
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```
使用以下方法处理NuScenes数据集：
```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```
对NuScenes数据集进行训练：
```bash
cd tools
bash scripts/dist_train.sh 1 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml
```
对NuScenes数据集进行测试：
```bash
cd tools
bash scripts/dist_test.sh 1 --cfg_file ./cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt ./voxelnext_nuscenes_kernel1.pth
```

Waymo开放数据集是由Waymo发布的用于自动驾驶研究的广泛数据集，收集自多种复杂城市环境。该数据集包含了高分辨率的激光雷达点云、摄像头图像和其他传感器数据，提供了详细的注释，用于3D目标检测、轨迹预测和场景理解等任务。

对于Waymo数据集，将其处理为以下形式：
```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)
 
├── pcdet
├── tools
```
使用以下方法处理Waymo数据集：
```python
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```
对Waymo数据集进行训练：
```bash
cd tools
bash scripts/dist_train.sh 1 --cfg_file cfgs/waymo_models/cbgs_voxel0075_voxelnext.yaml
```





