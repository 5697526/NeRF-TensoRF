import json
import numpy as np
from pathlib import Path


def robust_convert(colmap_text_dir, output_path, image_scale=1.0):
    cameras = {}
    images = []

    # 读取 cameras.txt
    with open(Path(colmap_text_dir)/"cameras.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            try:
                cam_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                params = list(map(float, parts[4:]))
                cameras[cam_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
            except (IndexError, ValueError) as e:
                print(f"跳过错误的相机行: {line.strip()} | 错误: {e}")

    # 读取 images.txt
    with open(Path(colmap_text_dir)/"images.txt", "r") as f:
        current_line = 0
        for line in f:
            current_line += 1
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            try:
                # 尝试新格式 (COLMAP >=3.7)
                if len(parts) == 10 and parts[9].endswith(('.jpg', '.png')):
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    name = parts[9]

                # 尝试旧格式或其他变体
                elif len(parts) >= 9:
                    # 假设最后一部分是文件名
                    name = parts[-1]
                    camera_id = int(parts[-2])
                    nums = list(map(float, parts[:-2]))
                    if len(nums) >= 7:
                        qw, qx, qy, qz = nums[0:4]
                        tx, ty, tz = nums[4:7]
                    else:
                        raise ValueError("参数不足")

                else:
                    raise ValueError("未知格式")

                # 四元数转旋转矩阵
                R = np.array([
                    [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                    [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                    [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                ])
                T = np.array([tx, ty, tz])
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = T

                images.append({
                    "file_path": name,
                    "transform_matrix": pose.tolist()
                })

            except Exception as e:
                print(f"跳过第 {current_line} 行: {line.strip()} | 错误: {e}")
                continue

    if not cameras:
        raise ValueError("未找到有效的相机参数")

    # 使用第一个相机的参数
    cam = next(iter(cameras.values()))
    fl = cam["params"][0] * image_scale

    output = {
        "camera_angle_x": 2 * np.arctan(cam["width"] / (2 * fl)),
        "frames": images
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image_scale", type=float, default=1.0)
    args = parser.parse_args()
    robust_convert(args.colmap_text, args.out, args.image_scale)
