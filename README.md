# NeRF & TensorRF 物体重建与新视图合成

### 一、项目概述：
整合了基于神经辐射场（NeRF）的物体三维重建与新视图合成技术，特别集成了 TensoRF 加速方案，可实现高效的物体建模与渲染。项目支持从多角度图像输入到三维模型重建、新视图生成的完整流程，适用于科研实验与应用开发。

### 二、数据介绍
`images/`存放拍摄的物体多角度图像共80张，图像按拍摄顺序编号命名。

### 三、文件结构
```
├── COLMAP.bat                 # COLMAP批处理脚本，用于相机参数估计
├── images/                    # 存放物体多角度拍摄图像的目录
├── nerf-pytorch/              # 原版NeRF的Python实现目录
│   ├── run_nerf.py            # 原版NeRF训练与测试主脚本
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
├── colmap2nerf.py             # COLMAP结果转NeRF格式的脚本
├── transforms.json            # 存储相机参数与图像对应关系的文件
├── database.db                # COLMAP数据库
└── ...                        # 其他辅助文件（如测试脚本等）
```

### 四、使用方法


### 五、模型权重下载


