# NeRF & TensoRF 物体重建与新视图合成

### 一、项目概述：
整合了基于神经辐射场（NeRF）的物体三维重建与新视图合成技术，特别集成了 TensoRF 加速方案，可实现高效的物体建模与渲染。项目支持从多角度图像输入到三维模型重建、新视图生成的完整流程，适用于科研实验与应用开发。

### 二、数据介绍
`images/`存放拍摄的物体多角度图像共80张，视角覆盖 360° 水平方向，图像按拍摄顺序编号命名。

### 三、文件结构
```
├── COLMAP.bat                 # COLMAP批处理脚本，用于相机参数估计
├── images/                    # 存放物体多角度拍摄图像的目录
├── nerf-pytorch/              # 原版NeRF的Python实现目录
│   ├── run_nerf.py            # 原版NeRF训练与测试主脚本
│   ├── configs/               
│   │   └── test.txt           # 原版NeRF模型配置
│   ├── data/                  
│   │   └── nerf_llff_data/    
│   │       └── test/          # 测试数据存放目录（含测试图像与相机参数）
│   └── logs/                  # 原版NeRF训练日志与模型检查点存储目录
├── TensoRF/                   # TensoRF加速方案目录
│   ├── configs/               
│   │   └── your_object.txt    # TensoRF模型配置文件（需自定义物体参数）
│   ├── data/                  # TensoRF预处理数据或中间结果目录
│   ├── log/                   
│   │   └── your_object_exp/   # TensoRF实验专属日志与结果目录
│   └── train.py               # TensoRF训练主脚本
├── sparse/0/                  # 稀疏点云与相机位姿文件
├── colmap2nerf.py             # COLMAP结果转NeRF格式的脚本
├── transforms.json            # 存储相机参数与图像对应关系的文件
├── database.db                # COLMAP数据库
└── ...                        # 其他辅助文件（如测试脚本等）
```

### 四、使用方法
#### 1. 依赖安装
配置 CUDA 环境，进入项目`nerf-pytorch/`目录，运行以下代码通过pip安装Python依赖。
```
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
pip install -r requirements.txt

```
#### 2. 运行 COLMAP 处理

在 COLMAP 中依次执行：`Feature Extraction` → `Feature Matching` → `Sparse Reconstruction`，生成相机位姿与稀疏点云，处理结果储存在：

sparse/：稀疏点云与相机位姿文件

database.db：特征匹配数据库

#### 3. 转换数据格式
运行 colmap2nerf 脚本，获取包含相机参数、图像路径与位姿矩阵的transforms.json文件。
```
python TensoRF\dataLoader\colmap2nerf.py --colmap_db database.db   --text sparse\0  --out ./transforms.json
```

#### 4. 模型训练

**NeRF方法:**

进入`nerf-pytorch/`目录，运行以下命令进行模型训练。
```
python run_nerf.py --config configs/test.txt --spherify --no_ndc
```

**TensoRF方法:**

进入`TensoRF/`目录，运行以下命令进行模型训练。
```
python train.py  --config configs/your_object.txt
```
#### 5. 渲染环绕视频
**NeRF方法:**

运行`run_nerf.py`会自动渲染视频保存至nerf-pytorch\logs\fern_test。

**TensoRF方法:**

运行以下代码对训练好的权重模型TensoRF\log\your_object_exp\your_object_exp.th进行渲染，结果保存至TensoRF\log\your_object_exp。

```
python train.py  --config configs/your_object.txt --render_only 1 --render_test 1 --render_train 1
```

#### 6. 定量评估

对模型训练过程中产生的log文件进行tensorboard可视化即可获取训练过程中在训练集和测试集上的loss曲线，以及在测试集上的PSNR等指标。


### 五、模型权重与可视化结果下载

渲染图像环绕视频以及权重模型均存储在我的[网盘](https://pan.baidu.com/s/1LUCqkAfinvkrUpj7BcvYjA?pwd=pprh)：https://pan.baidu.com/s/1LUCqkAfinvkrUpj7BcvYjA?pwd=pprh 提取码: pprh
