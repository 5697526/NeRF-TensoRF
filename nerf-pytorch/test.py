import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from run_nerf_helpers import img2mse, mse2psnr
from load_llff import load_llff_data


def evaluate_checkpoints():
    # 基础配置
    basedir = "./logs"
    expname = "fern_test"
    checkpoint_steps = range(10000, 210000, 10000)  # 从10k到200k，间隔10k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建TensorBoard writer
    eval_log_dir = os.path.join(basedir,  expname, "eval_logs")
    os.makedirs(eval_log_dir,  exist_ok=True)
    writer = SummaryWriter(log_dir=eval_log_dir)

    # 加载测试数据
    print("Loading test dataset...")
    images, poses, bds, render_poses, i_test = load_llff_data(
        "./data/llff/fern", factor=8)
    hwf = poses[0, :3, -1]
    H, W, focal = int(hwf[0]), int(hwf[1]), hwf[2]
    K = np.array([[focal,  0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]])

    # 准备测试数据
    test_images = torch.Tensor(images[i_test]).to(device)
    test_poses = torch.Tensor(poses[i_test]).to(device)

    # 加载初始模型配置
    from config_parser import config_parser
    args = config_parser().parse_args(["--datadir", "./data/llff/fern",
                                      "--expname", expname])

    # 创建初始模型
    _, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_test.update({
        'near': 0.,
        'far': 1.,
        'perturb': False,
        'raw_noise_std': 0.
    })

    # 遍历所有checkpoint
    for step in checkpoint_steps:
        ckpt_path = os.path.join(basedir,  expname, f"{step:06d}.tar")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping...")
            continue

        print(f"\nEvaluating checkpoint at step {step}...")

        # 加载模型权重
        ckpt = torch.load(ckpt_path)
        render_kwargs_test['network_fn'].load_state_dict(
            ckpt['network_fn_state_dict'])
        if render_kwargs_test['network_fine'] is not None:
            render_kwargs_test['network_fine'].load_state_dict(
                ckpt['network_fine_state_dict'])

        # 在测试集上评估
        test_losses = []
        test_psnrs = []

        with torch.no_grad():
            for i in range(len(test_images)):
                # 渲染当前测试视图
                rgb, disp, acc, _ = render(
                    H, W, K, chunk=args.chunk,
                    c2w=test_poses[i][:3, :4],
                    **render_kwargs_test
                )

                # 计算指标
                loss = img2mse(rgb, test_images[i])
                psnr = mse2psnr(loss)

                test_losses.append(loss.item())
                test_psnrs.append(psnr.item())

        # 计算平均指标
        avg_loss = np.mean(test_losses)
        avg_psnr = np.mean(test_psnrs)

        # 写入TensorBoard
        writer.add_scalar('test/loss',  avg_loss, step)
        writer.add_scalar('test/psnr',  avg_psnr, step)

        print(
            f"Step {step}: Test Loss = {avg_loss:.4f}, Test PSNR = {avg_psnr:.2f} dB")

    writer.close()
    print("\nEvaluation completed! Results saved to:", eval_log_dir)


if __name__ == '__main__':
    # 确保这些函数/类可用
    from run_nerf import create_nerf, render, config_parser
    from load_llff import load_llff_data

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluate_checkpoints()
