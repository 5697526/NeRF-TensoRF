import os
import json
from tensorboard.backend.event_processing import event_accumulator


def tensorboard_log_to_json(log_path, output_json_path):
    """
    将TensorBoard日志文件内容转换为JSON格式并保存

    :param log_path: TensorBoard事件文件路径
    :param output_json_path: 输出JSON文件路径
    """
    # 加载事件文件
    ea = event_accumulator.EventAccumulator(log_path,
                                            size_guidance={
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.TENSORS: 0,
                                                event_accumulator.IMAGES: 0,
                                                event_accumulator.AUDIO: 0,
                                                event_accumulator.HISTOGRAMS: 0,
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                            })
    ea.Reload()

    # 获取所有可用的标签类型
    available_tags = ea.Tags()

    # 构建包含所有数据的字典
    log_data = {}

    # 提取标量数据（如果存在）
    if 'scalars' in available_tags:
        log_data['scalars'] = {}
        for tag in available_tags['scalars']:
            events = ea.Scalars(tag)
            log_data['scalars'][tag] = [
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ]

    # 提取张量数据（如果存在）
    if 'tensors' in available_tags:
        log_data['tensors'] = {}
        for tag in available_tags['tensors']:
            events = ea.Tensors(tag)
            log_data['tensors'][tag] = [
                {'step': e.step, 'tensor': e.tensor_proto.SerializeToString().hex(),
                 'wall_time': e.wall_time}
                for e in events
            ]

    # 提取图像数据（如果存在）
    if 'images' in available_tags:
        log_data['images'] = {}
        for tag in available_tags['images']:
            events = ea.Images(tag)
            log_data['images'][tag] = [
                {'step': e.step, 'width': e.width, 'height': e.height,
                 'encoded_image': e.encoded_image_string.hex(),
                 'wall_time': e.wall_time}
                for e in events
            ]

    # 提取音频数据（如果存在）
    if 'audio' in available_tags:
        log_data['audio'] = {}
        for tag in available_tags['audio']:
            events = ea.Audio(tag)
            log_data['audio'][tag] = [
                {'step': e.step, 'sample_rate': e.sample_rate,
                 'num_channels': e.num_channels, 'length_frames': e.length_frames,
                 'encoded_audio': e.encoded_audio_string.hex(),
                 'content_type': e.content_type, 'wall_time': e.wall_time}
                for e in events
            ]

    # 提取直方图数据（如果存在）
    if 'histograms' in available_tags:
        log_data['histograms'] = {}
        for tag in available_tags['histograms']:
            events = ea.Histograms(tag)
            log_data['histograms'][tag] = [
                {'step': e.step, 'wall_time': e.wall_time,
                 'min': e.histogram_value.min,
                 'max': e.histogram_value.max,
                 'num': e.histogram_value.num,
                 'sum': e.histogram_value.sum,
                 'sum_squares': e.histogram_value.sum_squares,
                 'bucket_limits': list(e.histogram_value.bucket_limit),
                 'bucket_counts': list(e.histogram_value.bucket)}
                for e in events
            ]

    # 提取压缩直方图数据（如果存在）
    if 'compressed_histograms' in available_tags:
        log_data['compressed_histograms'] = {}
        for tag in available_tags['compressed_histograms']:
            events = ea.CompressedHistograms(tag)
            log_data['compressed_histograms'][tag] = [
                {'step': e.step, 'wall_time': e.wall_time,
                 'compressed_histogram': e.compressed_histogram_values.SerializeToString().hex()}
                for e in events
            ]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 保存为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"TensorBoard日志已成功保存为JSON文件: {output_json_path}")
    print(f"包含的数据类型: {', '.join(log_data.keys())}")


if __name__ == "__main__":
    # 输入和输出路径
    log_path = r"D:\FDU\codefield\colmap\nerf-pytorch\logs\fern_test\logs\20250618-075249\events.out.tfevents.1750233169.featurize.3064.0"
    output_json_path = r"D:\FDU\codefield\colmap\nerf-pytorch\logs\fern_test\logs\20250618-075249\log_data.json"

    # 执行转换
    tensorboard_log_to_json(log_path, output_json_path)
