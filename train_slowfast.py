# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)


    # ------------------------ Following values are modified from the default config ------------------------
    # Modify dataset type and path
    cfg.dataset_type = 'VideoDataset'
    cfg.data_root = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/train'
    cfg.data_root_val = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/val'
    cfg.ann_file_train = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_train_video.txt'
    cfg.ann_file_val = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_val_video.txt'
    cfg.ann_file_test = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_val_video.txt'

    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

    cfg.data.test.type = 'VideoDataset'
    cfg.data.test.ann_file = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_val_video.txt'
    cfg.data.test.data_prefix = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/val/'
    cfg.data.test.pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])]

    cfg.data.train.type = 'VideoDataset'
    cfg.data.train.ann_file = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_train_video.txt'
    cfg.data.train.data_prefix = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/train/'
    cfg.data.train.pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])]

    cfg.data.val.type = 'VideoDataset'
    cfg.data.val.ann_file = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/poseml_val_video.txt'
    cfg.data.val.data_prefix = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/val/'
    cfg.data.val.pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])]


    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)
    # Modify num classes of the model in cls_head
    cfg.model.cls_head.num_classes = 10
    # We can use the pre-trained TSN model
    cfg.load_from = './checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = '/vtti/scratch/asonth/PoseML_data6sec_for_mmaction2/workdir_slowfast'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    # cfg.data.videos_per_gpu = cfg.data.videos_per_gpu // 16
    cfg.data.videos_per_gpu=1
    cfg.data.workers_per_gpu=2
    cfg.optimizer.lr = cfg.optimizer.lr / 8
    cfg.total_epochs = 100

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 5
    # We can set the log print interval to reduce the the times of printing log
    cfg.log_config.interval = 5

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Save the best
    cfg.evaluation.save_best='auto'

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')
    # -------------------------------------------------------------------------------------------------


    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None or args.gpus is not None:
        warnings.warn(
            'The Args `gpu_ids` and `gpus` are only used in non-distributed '
            'mode and we highly encourage you to use distributed mode, i.e., '
            'launch training with dist_train.sh. The two args will be '
            'deperacted.')
        if args.gpu_ids is not None:
            warnings.warn(
                'Non-distributed training can only use 1 gpu now. We will '
                'use the 1st one in gpu_ids. ')
            cfg.gpu_ids = [args.gpu_ids[0]]
        elif args.gpus is not None:
            warnings.warn('Non-distributed training can only use 1 gpu now. ')
            cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed, distributed=distributed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatible with val workflow,
        # we recommend you to use `--validate`
        assert not cfg.omnisource
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()