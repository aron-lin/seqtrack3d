"""
main.py
Created by zenn at 2021/7/18 15:08
Modified by Aron Lin at Jun 1  09:42:22 CST 2023
"""
import pytorch_lightning as pl
import argparse

# import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything


from datasets import get_dataset
from models import get_model

torch.set_float32_matmul_precision("high")

import matplotlib.pyplot as plt
import sys

import datetime
import time

def generate_log_folder_name(cfg):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    cfg_name = cfg['cfg'].split("/")[-1].replace(".yaml", "")
    folder_name = f"output/{time_str}-{cfg_name}-{cfg['tag']}"
    return folder_name

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--save_top_k', type=int, default=5, help='save top k checkpoints')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--cfg', type=str, help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')
    parser.add_argument('--tag', type=str, default="", help='an extra tag appended on output folder name')
    parser.add_argument('--seed', type=int, help='random_seed')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


cfg = parse_config()
if cfg.seed is not None:
    seed_everything(cfg.seed)
    
env_cp = os.environ.copy()

try:
    node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

    is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
    pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
    print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

    if int(local_rank) == int(world_size) - 1:
        print(cfg)
except KeyError:
    pass


if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(cfg, type=cfg.train_type, split=cfg.train_split)
    val_data = get_dataset(cfg, type='test', split=cfg.val_split)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True,drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)
    learningrate_callback = LearningRateMonitor(logging_interval="step")

    # init trainer
    trainer = pl.Trainer(devices=-1, accelerator='auto', max_epochs=cfg.epoch,
                         callbacks=[checkpoint_callback,learningrate_callback],
                         default_root_dir=generate_log_folder_name(cfg),
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         num_sanity_val_steps=0,
                         gradient_clip_val=cfg.gradient_clip_val,
                         fast_dev_run=False)
    # init model
    train_dataloader_length = len(train_loader) #用于设置OneCycle学习率
    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg,train_dataloader_length=train_dataloader_length)
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg,train_dataloader_length=train_dataloader_length)

    trainer.fit(net, train_loader, val_loader, ckpt_path=cfg.checkpoint)
else:
    test_data = get_dataset(cfg, type='test', split=cfg.test_split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)

    trainer = pl.Trainer(devices=-1, accelerator='auto', default_root_dir=generate_log_folder_name(cfg))

    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg)
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
    trainer.test(net, test_loader, ckpt_path=cfg.checkpoint)
