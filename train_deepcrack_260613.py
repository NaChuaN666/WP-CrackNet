from __future__ import division
from __future__ import print_function

# Base
import os
import os.path as osp
from tqdm import tqdm
import random
import importlib
import argparse
import logging
import pdb
import datetime

from matplotlib import pyplot as plt

# DL
import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

# Custom
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils
# from evaluation import eval_in_script
from evaluation_deepcrack import eval_in_script



class ValidateEveryNEpochs(Callback):
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # 检查当前 epoch 是否是 n_epochs 的倍数
        if (trainer.current_epoch + 1) % self.n_epochs == 0:
            trainer.run_evaluation()


if __name__ == '__main__':

    # print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
    # print(torch.cuda.is_available())  # 检查 CUDA 是否可用
    # print(torch.__version__)  # PyTorch 版本
    print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    categories_crack = ['road', 'crack']
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="voc12/deepcrack/img_train.txt", type=str)
    parser.add_argument("--val_list", default="voc12/deepcrack/img_test.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    # Augmentation
    parser.add_argument("--resize", default=[384,512], nargs='+', type=float)
    parser.add_argument("--crop", default=[256,256], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)  # --cj 用于颜色增强

    # Hyper-parameters
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--grid", default=16, type=int)
    parser.add_argument("--prob", default=20, type=int)
    parser.add_argument("--W", default=[1, 0.2, 0.8, 0.3], nargs='+', type=float)

    # Learning rate
    parser.add_argument("--lr", default=0.001, type=float)  # 基准学习率不需要变
    parser.add_argument("--lr_seg", default=0.001, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)  # 权重衰减也不需要变
    parser.add_argument("--max_epochs", default=200, type=int)

    # Experiments
    # parser.add_argument("--exp", default='final', type=str)\
    parser.add_argument("--exp", default='recon_cvpr23_deepcrack_3_grid_collo_6_newval_SEPath_CECCM_used', type=str)
    # parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--name", default='deepcrack_PAAM_CECCM_used', type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=4242, type=int)
    parser.add_argument("--phase", default='train', type=str)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(6101)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    train_dataset = utils.build_dataset_deepcrack(args, phase='train', path=args.train_list)
    val_dataset = utils.build_dataset_deepcrack(args, phase='val', path=args.val_list)
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)
    
    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.max_epochs
    print("max_step is:", max_step)
    # exit()
    args.max_step = max_step

    dt = datetime.datetime.now()
    name_string = str(dt.month).zfill(2) + str(dt.day).zfill(2) + '_' + args.name

    logger = TensorBoardLogger('./experiments_deepcrack/', name=name_string, default_hp_metric=True)
    os.makedirs(logger.log_dir+'/dict', exist_ok=True)


    # 保存验证集最好的模型
    best_checkpoint_callback = ModelCheckpoint(monitor='val_miou',  # 监控验证集上的 mIoU
                                               save_top_k=1,  # 只保存最好的一个 checkpoint
                                               mode='max')  # mIoU 越大越好

    # 每 50 轮保存一次模型
    every_10_epochs_checkpoint_callback = ModelCheckpoint(save_top_k=-1,  # 保存所有的 checkpoint
                                                          save_last=False,  # 保持最后一个 checkpoint
                                                          period=100)  # 每 50 轮保存一次

    # 每轮验证
    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=1, logger=logger, callbacks=[best_checkpoint_callback, every_10_epochs_checkpoint_callback], check_val_every_n_epoch=1, progress_bar_refresh_rate=100)
    model = getattr(importlib.import_module('models.exp_'+args.exp), 'Exp')(args)
    # 输出exp_recon_cvpr23.py中的Exp类
    trainer.fit(model, train_data_loader, val_data_loader)

    # trainer.fit 会自动调用模型的 training_step 和 validation_step 方法
    # 在每个 epoch 内,训练阶段：遍历 train_data_loader 中的所有批次。调用 training_step 方法计算损失并更新模型参数
    # 在每个 epoch 内,验证阶段：在训练阶段结束后，遍历 val_data_loader 中的所有批次。调用 validation_step 方法计算验证指标