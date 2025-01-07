"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.image_datasets import load_data, save_combined_images
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        data_shuffle=args.data_shuffle,
    )
    logger.log("sampling...")
    
    num_samples = 0
    psnr_list = []  # 用于记录每张图片每个通道的 PSNR

    image_idx = 0  # 图像计数
    output_dir = os.path.join(args.log_dir, "combined_images")  # 保存拼接图像的目录
    while len(psnr_list) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        model_input, GT_7T = get_data(val_data)
        # print(f'GT_7T.device:{GT_7T.device}')
        # print(f'model.device:{model.device}')
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            input_tensor=model_input,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        total_psnr = [0.0, 0.0, 0.0]  # 每个通道的累计 PSNR

        for i in range(sample.shape[0]):
            generated_img = sample[i].cpu().numpy()
            gt_img = ((GT_7T[i] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
            input_img = ((model_input[i] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()

            # 保存拼接后的图像，并计算每个通道的 PSNR
            psnr_values = save_combined_images(input_img, gt_img, generated_img, output_dir, image_idx)
            image_idx += 1

            # 累加每个通道的 PSNR
            for channel in range(len(psnr_values)):
                total_psnr[channel] += psnr_values[channel]

            num_samples += 1
            psnr_list.append(psnr_values)

            # 输出每张图片每个通道的 PSNR
            logger.log(f"Image {image_idx}: PSNR per channel = {psnr_values}")

            # 计算每个通道的平均 PSNR
        avg_psnr = [total / (sample.shape[0]) for total in total_psnr]
        logger.log(f"Batch: Average PSNR per channel: {avg_psnr}")

        # logger.log(f"Samples saved to file. Sampling complete!")
    avg_psnr_all = np.mean(psnr_list, axis=0).tolist()
    print(f"all Average PSNR per channel: {avg_psnr_all}")
    logger.log(f"all Average PSNR per channel: {avg_psnr_all}")
    #     gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    #     all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    #     if args.class_cond:
    #         gathered_labels = [
    #             th.zeros_like(classes) for _ in range(dist.get_world_size())
    #         ]
    #         dist.all_gather(gathered_labels, classes)
    #         all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    #     logger.log(f"created {len(all_images) * args.batch_size} samples")
    #
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def get_data(data):
    batch = next(data)
    batch_3 = batch['IN']['data'].squeeze(1)  # 去掉多余的维度
    batch_7 = batch['GT']['data'].squeeze(1)
    if isinstance(batch_3, th.Tensor):
        # 如果是 PyTorch 的 Tensor
        # print("Data is a PyTorch Tensor")
        # 将 [B, H, W, C] 转为 [B, C, H, W]
        batch_3 = batch_3.permute(0, 3, 1, 2)
        batch_7 = batch_7.permute(0, 3, 1, 2)
    elif isinstance(batch_3, np.ndarray):
        # 如果是 NumPy 的 ndarray
        # print("Data is a NumPy array")
        # 将 [B, H, W, C] 转为 [B, C, H, W]
        batch_3 = np.transpose(batch_3, (0, 3, 1, 2))
        batch_7 = np.transpose(batch_7, (0, 3, 1, 2))
    else:
        raise TypeError("Unsupported data type. Expected torch.Tensor or numpy.ndarray.")

    return batch_3, batch_7

def create_argparser():
    defaults = dict(
        log_dir="",
        data_dir="",
        patch_size=(256,256,3),
        patch_overlap=(0,0,2),
        clip_denoised=True,
        num_samples=500,
        batch_size=1,
        use_ddim=False,
        model_path="",
        data_shuffle=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
