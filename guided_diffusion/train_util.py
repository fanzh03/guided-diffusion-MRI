import copy
import functools
import os
import numpy as np
import random
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            tb_logger,
            path,
            data,
            # val_data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,

    ):
        self.model = model
        self.diffusion = diffusion
        self.path = path
        self.data = data
        # self.val_data = val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.tb_logger = tb_logger
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def get_data(self):
        batch = next(self.data)
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
         # 遍历 batch 中的每个样本，按 p=0.1 的概率将其替换为全黑图
        for i in range(batch_3.shape[0]):
            if random.random() < 0.05:  # 对每个样本有 5% 的概率替换
                if isinstance(batch_3, th.Tensor):
                    batch_3[i] = th.zeros_like(batch_3[i])  # 替换为全零
                    batch_7[i] = th.zeros_like(batch_7[i])
                elif isinstance(batch_3, np.ndarray):
                    batch_3[i] = np.zeros_like(batch_3[i])
                    batch_7[i] = np.zeros_like(batch_7[i])

        return batch_3, batch_7, {}

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch_3, batch_7, cond = self.get_data()
            self.run_step(batch_3, batch_7, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # self.validate(self.val_data)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_3, batch_7, cond):
        self.forward_backward(batch_3, batch_7, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch_3, batch_7, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch_7.shape[0], self.microbatch):
            micro = batch_7[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch_7.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                batch_3,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.tb_logger.add_scalar("train_mse_loss", loss, self.step)
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    # def validate(self, data):
    #     total_mse_loss = 0.0
    #     psnr_loss_in = 0.0
    #     psnr_loss_out = 0.0
    #     total_samples = 0
    #     self.model.eval()  # 切换到评估模式
    #     with th.no_grad():  # 禁用梯度计算
    #         for i in range(0, data.shape[0], self.microbatch):
    #             micro_gt = data[i: i + self.microbatch].to(dist_util.dev())
    #             micro_in = data[i: i + self.microbatch].to(dist_util.dev())
    #             t, weights = self.schedule_sampler.sample(micro_gt.shape[0], dist_util.dev())

    #             compute_losses = functools.partial(
    #                 self.diffusion.validation,
    #                 self.ddp_model,
    #                 micro_in,
    #                 micro_gt,
    #                 t,
    #                 self.step,
    #                 os.path.join(self.path, r'val/mid_result')
    #             )

    #             losses = compute_losses()
    #             loss = (losses["loss"] * weights).mean()
    #             total_mse_loss += loss.item() * micro_gt.shape[0]
    #             total_samples += self.microbatch

    #     mse_loss = total_mse_loss / total_samples
    #     logger.logkv(f"val_l1", mse_loss)  # stdout输出Loss
    #     self.tb_logger.add_scalar("val_l1_loss", mse_loss, self.step)
    #     self.model.train()  # 切换回训练模式

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
