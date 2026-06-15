import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import importlib
import argparse
import logging
import tools.utils as utils
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from networks import resnet38d
from torch import nn
import matplotlib.pyplot as plt
import cv2

def main():
     # 加载训练好的模型
     parser = argparse.ArgumentParser()

     # Dataset
     parser.add_argument("--train_list", default="voc12/deepcrack_237/img_train.txt", type=str)
     parser.add_argument("--val_list", default="voc12/deepcrack_237/infer_237.txt", type=str)
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
     parser.add_argument("--lr_seg", default=0.001, type=float)  # 基准学习率不需要变
     parser.add_argument("--wt_dec", default=5e-4, type=float)  # 权重衰减也不需要变
     parser.add_argument("--max_epochs", default=200, type=int)

     # Experiments
     # parser.add_argument("--exp", default='final', type=str)\
     parser.add_argument("--exp", default='recon_cvpr23_deepcrack_3_grid_collo_6_newval_SEPath_CECCM_used', type=str)
     # parser.add_argument("--name", required=True, type=str)
     # /root/autodl-tmp/ACR_used_boundary/experiments_deepcrack//version_3/checkpoints
     parser.add_argument("--name", default='0613_deepcrack_PAAM_CECCM_used', type=str)
     parser.add_argument("--gpu", default=-1, type=int)
     parser.add_argument("--seed", default=4242, type=int)
     parser.add_argument("--phase", default='train', type=str)

     args = parser.parse_args()

     device = 'cuda'

     train_dataset = utils.build_dataset_deepcrack(args, phase='train', path=args.train_list)
     val_dataset = utils.build_dataset_deepcrack(args, phase='val', path=args.val_list)
     
     train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
     val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)
     
     train_num_img = len(train_dataset)
     train_num_batch = len(train_data_loader)
     max_step = train_num_img // args.batch_size * args.max_epochs
     args.max_step = max_step

     model = getattr(importlib.import_module('models.exp_'+args.exp), 'Exp')(args)
     # 选用checkpoint
     dict_dir = os.path.join('./experiments_deepcrack', args.name, 'version_1', 'checkpoints', 'epoch=168-step=25180.ckpt')
     # checkpoint = torch.load(dict_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')   
     model.load_pretrained_seg(dict_dir)
     # model.load_state_dict(checkpoint['state_dict'])

     # Move model to GPU if available
     if torch.cuda.is_available():
          model = model.to('cuda')

     output_dir = os.path.join('./experiments_deepcrack/', args.name, 'version_1/out_ck168')
     if not os.path.exists(output_dir):
          os.makedirs(output_dir)
          
     with torch.no_grad():
          for name, img_list, lab in val_data_loader:
               print(name)
               image_name = name[0]    
               img = img_list[2] # (1,3,256,256)
               lab = lab
               seg_mask = model.seg_test_recon_0416(img.to(device))
               output_path = os.path.join(output_dir, '%s.png' % image_name)
               plt.imsave(output_path, seg_mask, cmap='gray')
     
     print("no error to this")
     exit()


     # 测试图像所在的目录
     eval_list = 'infer_226_name'
     test_name_txt = './voc12/deepcrack/infer_226_name.txt'
     df = pd.read_csv(test_name_txt, names=['filename'])
     name_list = df['filename'].values

     test_images_dir = './data/Deepcrack/images_all'
     transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据你的模型输入大小调整
            transforms.ToTensor(),
        ])
     
     device = 'cuda'

     output_dir = os.path.join('./experiments_deepcrack/', args.name, 'version_1/out_ck85')
     if not os.path.exists(output_dir):
          os.makedirs(output_dir)

     # 遍历测试图像并进行测试
     for image_name in name_list:
          image_path = os.path.join(test_images_dir, '%s.png'%image_name)
          image = Image.open(image_path).convert('RGB')
          image_tensor = transform(image).unsqueeze(0).to(device) # torch.Size([1, 3, 256, 256])
          # seg_mask = model.seg_test(image_tensor)
          seg_mask = model.seg_test_recon(image_tensor)
          seg_mask = seg_mask * 255
          # 保存图像到指定文件夹
          output_path = os.path.join(output_dir, '%s.png' % image_name)
          # plt.imsave(output_path, seg_mask, cmap='gray')
          cv2.imwrite(output_path, seg_mask)

          # seg_mask_image.save(output_path)

     print("all outputs have been saved!")
if __name__ == '__main__':
    main()