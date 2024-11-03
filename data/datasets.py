import logging
import random
import time

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate

from data.kubric import KubricDataloader
from data.kinetics import KineticsDataset

logger = logging.getLogger(__name__)

def collate_fn(batch):
    return default_collate(batch)


def make_data_sampler(local_rank, dataset, shuffle=True):
    torch.manual_seed(0)
    if local_rank in [-2, -1]:
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

    return sampler


def get_data_loader(args, training, val_during_train=False):
    st = time.time()

    IMG_SIZE1 = (args.img_size1_h, args.img_size1_w)
    IMG_SIZE2 = (args.img_size2_h, args.img_size2_w)

    traindir = args.data_path

    if args.dataset_type == "kinetics":
        traindir = "datasets/kinetics700-2020"
        dataset = KineticsDataset(
            use_frame_transform=True,
            img_size1=IMG_SIZE1,
            img_size2=IMG_SIZE2,
            split="train",
            root=traindir,
            no_of_frames=args.no_of_frames,
            random_seed=args.seed,
            aug_setting=args.data_aug_setting,
            training=training,
        )

        logger.info(f"Dataset loading took {str(time.time() - st)}")

        logger.info("Creating data loaders")
        sampler = make_data_sampler(args.local_rank, dataset, shuffle=training)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.eval_batch_size
            if args.eval_only
            else args.train_batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=random.seed(args.seed),
        )

    elif args.dataset_type == "kubric":
        split = "train" if training else "validation"
        batch_size = args.eval_batch_size if args.eval_only else args.train_batch_size

        if args.local_rank not in [-2, -1]:
            args.world_size = dist.get_world_size()
        else:
            args.world_size = -1

        if args.random_frame_skip:
            frame_len_diff = -1
        else:
            frame_len_diff = 1
        
        data_loader = KubricDataloader(
            use_frame_transform=True,
            img_size1=IMG_SIZE1,
            img_size2=IMG_SIZE2,
            split=split,
            root=traindir,
            no_of_frames=args.no_of_frames,
            frame_len_diff=frame_len_diff,
            random_frame_skip=args.random_frame_skip,
            aug_setting=args.data_aug_setting,
            random_seed=args.seed,
            batch_size=batch_size,
            shuffle=False,
            worker_id=args.local_rank,
            num_workers=args.world_size,
            num_parallel_point_extraction_calls=args.workers,
            training=training,
        )

        logger.info(f"Dataset loading took {str(time.time() - st)} secs")
    else:
        raise NotImplementedError

    return data_loader, args
