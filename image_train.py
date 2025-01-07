"""
Train a diffusion model on images.
"""

import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import sys
sys.path.append('..')  # 将项目根目录添加到 Python 路径中

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# torch.cuda.empty_cache()  # 手动释放未使用显存
def main():
    args = create_argparser().parse_args()
    tb_logger = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir, format_strs=["stdout", "log"])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    logger.log(f"总参数量: {total_params:,}")
    print(f"patch_size: {args.patch_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"log_dir: {args.log_dir}")
    # print(f"可训练参数量: {trainable_params:,}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        data_shuffle=args.data_shuffle,
    )
    
    # val_data = load_data(
    #     data_dir=args.data_val_dir,
    #     batch_size=args.batch_size,
    #     patch_size=args.patch_size,
    #     patch_overlap=args.patch_overlap,
    #     deterministic=True,
    # )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        path=args.log_dir,
        data=data,
        # val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        tb_logger=tb_logger,
    ).run_loop()


def create_argparser():
    defaults = dict(
        log_dir="",
        data_dir="",
        # data_val_dir="",
        patch_size=(256,256,3),
        patch_overlap=(8,8,0),
        batch_size=1,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_shuffle=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
