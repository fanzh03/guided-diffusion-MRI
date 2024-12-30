import math
import os
import random
import torchio as tio
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
        *,
        data_dir,
        batch_size,
        patch_size,
        patch_overlap,
        max_queue_length=10000,
        num_workers=0,
        deterministic=False,
        # class_cond=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param patch_overlap: patch_overlap 重叠区域大小
    :param num_workers: num_workers
    :param max_queue_length: max_queue_length
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    # :param class_cond: if True, include a "y" key in returned dicts for class
    #                    label. If classes are not available and this is true, an
    #                    exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    # :param random_crop: if True, randomly crop the images for augmentation.
    # :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    subjects = load_paired_subjects(data_dir)

    # 构建 TorchIO SubjectsDataset
    dataset = tio.SubjectsDataset(subjects)
    # 定义采样器
    patch_size = patch_size
    patch_overlap = patch_overlap
    sampler = tio.data.GridSampler(subject=dataset[0],  # 使用第一个样本定义采样器
                                   patch_size=patch_size,
                                   patch_overlap=patch_overlap)
    num_patches = len(sampler)
    # aggregator = tio.inference.GridAggregator(sampler, 'hann')

    # 定义 patch 队列
    max_queue_length = max_queue_length
    num_workers = num_workers
    patches = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_queue_length,
        samples_per_volume=num_patches,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_patches=True,
    )

    # 构建 DataLoader
    if deterministic:
        patch_loader = DataLoader(
            patches,
            batch_size=batch_size,
            pin_memory=False,  # False
            shuffle=False,
        )
    else:
        patch_loader = DataLoader(
            patches,
            batch_size=batch_size,
            pin_memory=False,  # False
            shuffle=True,
        )
    while True:
        yield from patch_loader


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def load_paired_subjects(fp3):
    """
    加载配对的 3T 和 7T 图像数据，组织为监督学习 IN-GT 格式。
    Args:
        fp3 (str): 3T 图像文件所在目录。
        # fp7 (str): 7T 图像文件所在目录。
        # postfix (str): 文件名后缀（如 ".nii.gz",".nii"）。
    Returns:
        list: 配对的 TorchIO Subject 对象列表，每个 Subject 包含 IN（3T）和 GT（7T）。
    """
    subjects = []
    postfix = '.nii'
    # 获取 3T 文件名列表
    filenames_3T = [f for f in os.listdir(fp3) if f.endswith(postfix)]
    parent_dir = os.path.dirname(fp3)
    fp7 = os.path.join(parent_dir, "7T")

    for filename in filenames_3T:
        path_3T = os.path.join(fp3, filename)
        path_7T = os.path.join(fp7, filename)

        # 检查配对的 7T 图像是否存在
        if not os.path.exists(path_7T):
            print(f"Warning: Missing paired 7T image for {filename}. Skipping.")
            continue

        # 加载并归一化图像
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        t1_3T = rescale(tio.ScalarImage(path_3T))
        t1_7T = rescale(tio.ScalarImage(path_7T))
        subject = tio.Subject(
            IN=t1_3T,  # 输入图像 (3T)
            GT=t1_7T,  # 目标图像 (7T)
            # id=filename
        )
        subjects.append(subject)

    print(f"Loaded {len(subjects)} paired subjects.")
    return subjects


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]
