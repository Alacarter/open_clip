import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from webdataset.utils import identity
import webdataset as wds

from clip.clip import (
    get_tokenize_fn, clip_tokenize, custom_tokenize
)


class CsvDataset(Dataset):
    def __init__(
            self, input_filename, transforms, img_key, caption_key,
            tokenize_scheme, sep=","):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        print(df.columns)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize_fn = get_tokenize_fn(tokenize_scheme)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize_fn([str(self.captions[idx])])[0]
        return images, texts

class CsvDatasetTwoTransforms(Dataset):
    def __init__(self, input_filename, teacher_transforms, student_transforms, img_key, caption_key, sep=","):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        print(df.columns)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.teacher_transforms = teacher_transforms
        self.student_transforms = student_transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        teacher_transformed_images = self.teacher_transforms(Image.open(str(self.images[idx])))
        teacher_texts = clip_tokenize([str(self.captions[idx])])[0]
        student_transformed_images = self.student_transforms(Image.open(str(self.images[idx])))
        student_texts = custom_tokenize([str(self.captions[idx])])[0]
        return teacher_transformed_images, teacher_texts, student_transformed_images, student_texts

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text, tokenize_scheme):
    tokenize_fn = get_tokenize_fn(tokenize_scheme)
    return tokenize_fn([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes = eval(open(os.path.join(dir_path, 'sizes.json'), 'r').read())
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_wds_dataset(args, preprocess_img, is_train, tokenize_scheme):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    # The following code is adapted from https://github.com/tmbdev/webdataset-examples/blob/master/main-wds.py
    num_samples, num_shards = get_dataset_size(input_shards)
    if is_train and args.distributed:
        max_shards_per_node = math.ceil(num_shards / args.world_size)
        num_samples = args.world_size * (num_samples * max_shards_per_node // num_shards)
        num_batches = num_samples // (args.batch_size * args.world_size)
        num_samples = num_batches * args.batch_size * args.world_size
    else:
        num_batches = num_samples // args.batch_size
    shardlist = wds.PytorchShardList(
        input_shards,
        epoch_shuffle=is_train,
        split_by_node=is_train  # NOTE: we do eval on a single gpu.
    )
    preprocess_txt_fn = lambda x: preprocess_txt(x, tokenize_scheme)
    dataset = (
        wds.WebDataset(shardlist)
        .decode("pil")
        .rename(image="jpg;png", text="txt")
        .map_dict(image=preprocess_img, text=preprocess_txt_fn)
        .to_tuple("image", "text")
        .batched(args.batch_size, partial=not is_train or not args.distributed)
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
    )
    if is_train and args.distributed:
        # With DDP, we need to make sure that all nodes get the same number of batches;
        # we do that by reusing a little bit of data.
        dataloader = dataloader.repeat(2).slice(num_batches)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)

def get_csv_dataset(args, preprocess_fn, is_train, tokenize_scheme):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        tokenize_scheme=tokenize_scheme,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_distillation_data(args, preprocess_fns, is_train):
    teacher_preprocess_fn, student_preprocess_fn = preprocess_fns
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    ext = input_filename.split('.')[-1]
    assert ext == "csv", "Unsupported data_path extension."
    dataset = CsvDatasetTwoTransforms(
        input_filename,
        teacher_preprocess_fn,
        student_preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.distillation:
        if args.train_data:
            data["train"] = get_distillation_data(args, preprocess_train, is_train=True)
        if args.val_data:
            data["val"] = get_distillation_data(args, preprocess_val, is_train=False)
    else:
        if args.train_data:
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, tokenize_scheme=args.tokenize_scheme)
        if args.val_data:
            data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
                args, preprocess_val, is_train=False, tokenize_scheme=args.tokenize_scheme)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
