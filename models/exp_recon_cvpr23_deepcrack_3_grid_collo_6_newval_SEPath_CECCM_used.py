from cmath import isnan
import random
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import exp
from torch.autograd import Variable

import os

import matplotlib.pyplot as plt
import cv2

import pytorch_lightning as pl

from img import crf_inference_ysh

# Image tools
from matplotlib import pyplot as plt

from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image, _crf_with_alpha_deepcrack
# from evaluation_precision import eval_in_script
from evaluation_precision_deepcrack import eval_in_script, eval_in_script_CRF, eval_seg

# import resnet38d
from networks import resnet38d

from models.msgms_loss import MSGMS_Loss
from torch.nn import BCEWithLogitsLoss

from modules import cam_consistency_loss, edge_consistency_loss, center_enhance, normalize
import tools.imutils as imutils


# SSIM:
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class Exp(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.check_sanity = True

        self.args = args
        self.categories = ['road', 'crack']

        # Hyper-params
        self.grid = args.grid
        self.prob = args.prob
        self.W = args.W     # [1, 0.2, 0.8, 0.3]

        # Loss & Metric
        self.bce = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        self.max_miou = 0
        self.right_count = 0
        self.wrong_count = 0

        self.build_framework(args.phase)

    def build_framework(self, phase):

        args = self.args

        # Network
        self.net_recon = resnet38d.Net_recon(args.D)

        self.net_recon.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)

        self.net_main = resnet38d.Net_main(args.D)

        self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)

        
        self.seg_decoder = resnet38d.SimpleSegDecoder_fusion_SEPath(input_channels1=1024, input_channels2=1024, num_classes=1)

        if phase=='train':

            print('define optimizers')
            # Optimizer
            param_recon = self.net_recon.get_parameter_groups()
            self.opt_recon = utils.PolyOptimizer([
                {'params': param_recon[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
                {'params': param_recon[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
                {'params': param_recon[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
                {'params': param_recon[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
            ],
                lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

            param_main = self.net_main.get_parameter_groups()
            self.opt_main = utils.PolyOptimizer([
                {'params': param_main[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
                {'params': param_main[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
                {'params': param_main[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
                {'params': param_main[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
            ],
                lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

            lr_seg = args.lr_seg*0.99**(self.current_epoch)
            self.opt_seg = torch.optim.AdamW(self.seg_decoder.parameters(), lr=lr_seg)
            for param_group in self.opt_seg.param_groups:
                param_group['lr'] = lr_seg


    def training_step(self, batch, batch_idx):
        
        tb = self.logger.experiment
        img = batch['img']  # B x 3 x H x W
        
        name = batch['name']
        label = batch['label']

        img_denorm_rgb_hwc = batch['img_rgb_hwc']

        B = img.shape[0] # batch = 4
        H = img.shape[2]
        W = img.shape[3]
        
        C = 2  # Number of cls

        # 初始化losses:
        loss_recon = 0.0  # 默认值
        loss_recon_cls = 0.0
        loss_recon_full = 0.0
        loss_recon_t = 0.0
        loss_recon_nt = 0.0
        loss_main = 0.0
        loss_main_cls = 0.0
        loss_main_t = 0.0
        loss_main_nt = 0.0

        # 判断哪些图像是纯背景，哪些是裂痕
        mask_background = (label[:, 0] == 1) # 纯背景图像
        mask_crack = ~mask_background  # 包含裂痕的图像

        label_t, label_nt = self.split_label(label, B) # t:target, nt:non-target
        # class_t is tensor([1, 0, 1, 1], device='cuda:0')
        class_t = label_t.nonzero()[:,1] # 获取目标类的索引

        epoch_flag = 0
        if epoch_flag == 0:  # 训练分类器和生成器阶段
            ################################### Train the recon network ###################################

            self.opt_recon.zero_grad() 
            self.net_recon.train()
            self.net_main.eval()  # classifier is locked!

            # 生成随机掩码, 用于屏蔽特定区域
            mask_small = torch.randint(0, 100, size=(B,1,self.grid,self.grid))<self.prob + self.current_epoch
            mask_grid = F.interpolate(mask_small.float(), size=[H,W], mode='nearest').cuda(self.device) # For VRF

            with torch.no_grad():
                # 提取主网络 (net_main) 的类别激活图 (CAM)
                cam_small, _, feat_from_main, _ = self.net_main(img)
                # cam_small = self.max_norm(cam_small)*label.view(B,20,1,1)
                cam_small = self.max_norm(cam_small)*label.view(B,2,1,1)  # 对 CAM 进行最大值归一化 (0~1)
                cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False) 

                #  构造目标 (target) 和非目标 (non-target) 的语义区域
                # cam.shape: torch.Size([4, 2, 256, 256])
                cam_t = cam[label_t==1].unsqueeze(1) # 提取目标类别的区域
                # then cam_t.shape: torch.Size([4, 1, 256, 256])

                segment_t_fuse_rev = cam_t.float()*mask_grid
                segment_t_fuse = 1 - segment_t_fuse_rev # Assuring virtual remnant  # 生成目标区域的辅助掩码

                cam_nt = 1-cam_t   # 非目标区域 
                segment_nt_fuse_rev = cam_nt*mask_grid
                segment_nt_fuse = 1-segment_nt_fuse_rev # Assuring virtual remnant  # 生成非目标区域的辅助掩码

                # 用于检查目标区域是否覆盖了足够的像素点，避免过小的区域对训练无意义
                valid = (torch.sum(segment_t_fuse.view(B,-1),dim=-1))>50

            # 重建特征生成
            cam_small, pred, feat_small, _ = self.net_recon(img)  # 将输入图像送入 重建网络 提取原始特征
            feat = F.interpolate(feat_small, size=[H,W], mode='bilinear', align_corners=False) # feat.shape is torch.Size([4, 448, 256, 256])
            # 特征被分为两部分：目标特征 (feat_masked_t)：分离出目标区域的特征; 非目标特征 (feat_masked_nt)：分离出非目标区域
            
            feat_masked_t = feat * segment_t_fuse  # 只对目标区域进行特征聚焦
            feat_masked_nt = feat * segment_nt_fuse # 聚焦非目标区域

            # no-color
            # 重建图像生成
            img_recon_full = self.net_recon.recon_decoder(feat.detach(), use_tanh=False) # For decoder only  #从完整特征重建 
            img_recon_from_t = self.net_recon.recon_decoder(feat_masked_t, use_tanh=False)  # 从目标区域特征重建
            img_recon_from_nt = self.net_recon.recon_decoder(feat_masked_nt, use_tanh=False) # 从非目标区域特征重建

            loss_recon_cls = self.bce(pred,label)   # 分类损失

            if mask_crack.any():
                loss_recon_full = self.L1(img_recon_full[mask_crack], img[mask_crack]) # 全图重建的 L1 损失 
                loss_recon_t = self.masked_L1(img_recon_from_t[mask_crack], img[mask_crack], segment_t_fuse_rev[mask_crack], valid[mask_crack]) # 针对目标区域的 L1 损失
                loss_recon_nt = self.masked_L1(img_recon_from_nt[mask_crack], img[mask_crack], segment_nt_fuse_rev[mask_crack])  # 针对非目标区域的 L1 损失
            
            # self.W:  [1, 0.2, 0.8, 0.3]
            loss_recon = self.W[0]*loss_recon_cls + self.W[1]*loss_recon_full + 0.3*loss_recon_t + 0.5*loss_recon_nt
            loss_recon.backward()

            if torch.isnan(loss_recon):
                print('reconNaN!')
            else:       
                self.opt_recon.step()

            ################################### Train the main network ###################################
            self.opt_main.zero_grad() # 清除主网络参数的梯度
            self.net_main.train()   # 将主网络设置为训练模式
            self.net_recon.eval()  # recon_encoder and recon_decoder are locked! # 将重建网络设置为评估模式

            with torch.no_grad(): # 冻结重建网络，避免生成梯度
                cam_small_recon, _, feat_small, _ = self.net_recon(img) # 提取特征 (低分辨率)
                cam_recon = self.max_norm(cam_small_recon) * label.view(B, 2, 1, 1)
                cam_recon = F.interpolate(cam_recon, size=[H, W], mode='bilinear', align_corners=False)  # [4,2,256,256]
                feat = F.interpolate(feat_small, size=[H,W], mode='bilinear', align_corners=False)  # 调整到图像大小

            cam_small, pred, feat_from_main, _ = self.net_main(img)  # 主网络的前向传播
            # argmax 获取分割预测，结果是每个像素点被预测为哪个类别
            seg_pred = torch.argmax(F.interpolate(cam_small,(H,W),mode='bilinear',align_corners=False), dim=1).unsqueeze(1)


            # 调用 max_norm 对 cam_small 归一化（范围为 0 到 1）
            # 根据标签 (label) 过滤，只保留和当前任务相关的类别
            cam_small = self.max_norm(cam_small)*label.view(B,2,1,1)
            cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False)

            segment_t = cam[label_t==1].unsqueeze(1) # Soft # 提取目标区域 (Soft掩码)
            segment_nt = 1-segment_t # 提取非目标区域

            feat_t = feat.detach() * segment_t # 从特征中提取目标区域 Train main network only, while freezing recon network.
            feat_nt = feat.detach() * segment_nt # 从特征中提取非目标区域

            # 使用locked重建网络的解码器 (recon_decoder) 从目标和非目标特征解码生成图像
            img_recon_from_t = self.net_recon.recon_decoder(feat_t, use_tanh=False) # 目标区域生成重建图像
            img_recon_from_nt = self.net_recon.recon_decoder(feat_nt, use_tanh=False)  # 非目标区域生成重建图像


            # 构造评估掩码
            #  segment_t==0 : 目标区域的像素位置为 True
            #  .float() 将 布尔张量转换为浮点数张量。True 转换为 1.0，False 转换为 0.0
            eval_mask_for_t = (segment_t==0).float().detach() # Hard # 硬掩码：目标区域外的像素,强调目标区域外部的像素点
            # 软掩码：目标区域内匹配预测的区域,强调目标区域 内部，且分割预测正确的像素点
            eval_mask_for_nt = segment_t.clone().detach() * (seg_pred==class_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).float() # Soft

            loss_main_cls = self.bce(pred,label)  # 主分类的 BCE 损失 

            # 由目标(例如crack) feature 重建整个图像，计算和非目标区域(road)之间的损失，希望loss大，因为loss越大，说明classifier分的越正确
            # loss_main_t = -self.masked_L1(img_recon_from_t, img, eval_mask_for_t)  # 对目标区域外的重建损失
            if mask_crack.any():
                loss_main_t = -self.masked_L1(img_recon_from_t[mask_crack], img[mask_crack], eval_mask_for_t[mask_crack])
                loss_main_nt = -self.masked_L1(img_recon_from_nt[mask_crack], img[mask_crack], eval_mask_for_nt[mask_crack]) 
            

            # Step 1: 中心增强
            cam_norm = normalize(cam)
            cam_recon_norm = normalize(cam_recon)
            cam_t_main_enhanced = center_enhance(cam_norm)
            cam_t_recon_enhanced = center_enhance(cam_recon_norm)

            λ_consistency = 0.5 # 可调参数
            loss_cam_consistency = cam_consistency_loss(cam_t_main_enhanced[mask_crack], cam_t_recon_enhanced[mask_crack])

            # self.W:  [1, 0.2, 0.8, 0.3]
            loss_main = loss_main_cls + self.args.W[2]*loss_main_t + self.args.W[3]*loss_main_nt + λ_consistency * loss_cam_consistency
            loss_main.backward()

            if torch.isnan(loss_main):
                print('mainNaN!')
            else:       
                self.opt_main.step()

        if self.current_epoch > -1:
            ####################################################训练分割decoder
            set_recon_train = (self.current_epoch > 50 and self.current_epoch % 2 == 0)
            if set_recon_train:
                cam_small, pred, feat_from_main, feat_out_main = self.net_main(img)
                cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False)
                _, _, _, feat_out = self.net_recon(img) # 提取特征 (低分辨率)
            else:
                with torch.no_grad():
                    cam_small, pred, feat_from_main, feat_out_main = self.net_main(img)
                    cam = F.interpolate(cam_small, size=[H,W], mode='bilinear', align_corners=False)
                    _, _, _, feat_out = self.net_recon(img) # 提取特征 (低分辨率)
            loss_seg = None
            if mask_crack.any():
                cam_label = self.denseCRF(cam[mask_crack], label_t[mask_crack], img_denorm_rgb_hwc[mask_crack]) # [3,1,256,256]
                pos_weight = torch.tensor([5.0]).to(self.device)
                criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
                if cam_label.max() == 0:
                    print("Skipping step because cam_label max is 0.")
                else:
                    self.opt_seg.zero_grad()  # 清除分割decoder梯度
                    self.seg_decoder.train()  # 将分割decoder网络设置为训练模式

                    if set_recon_train:
                        self.opt_recon.zero_grad()  # 清除重建器的梯度
                        self.net_recon.train() # 将重建器设置为训练模式

                        self.opt_main.zero_grad()  # 清除分类器的梯度
                        self.net_main.train()   # 将主网络设置为训练模式

                        # 冻结分类器与重建器的decoder,即所使用的genUnet:
                        for param in self.net_recon.genUnet.parameters():
                            param.requires_grad = False

                    # feat_out是重建器编码器得到，feat_out_main是从分类器编码器得到

                    seg_output = self.seg_decoder(feat_out, feat_out_main)

                    loss_seg = criterion(seg_output[mask_crack], cam_label)
                    if loss_seg <= 0:
                        print("loss_seg has error as <=0")
                    loss_seg.backward()

                    if torch.isnan(loss_seg):
                        print('mainNaN!')
                    else:
                        
                        self.opt_seg.step()
                        if set_recon_train:
                            self.opt_recon.step()  # 更新重建器编码器的参数
                            self.opt_main.step() # 更新分类器编码器的参数
        ######################################################################################################################
        ####################################################### Export #######################################################
        ######################################################################################################################

        # 收集损失值
        losses = {}
        losses['loss_recon'] = loss_recon
        losses['loss_recon_cls'] = loss_recon_cls
        losses['loss_recon_full'] = loss_recon_full
        losses['loss_recon_mask'] = loss_recon_t
        losses['loss_main'] = loss_main
        losses['loss_main_cls'] = loss_main_cls
        losses['loss_main_t'] = loss_main_t
        losses['loss_main_nt'] = loss_main_nt

        # #
        # 调用 count_rw 方法来计算当前批次的预测正确数量和错误数量。
        # 该方法会更新模型的正确和错误预测计数，以便后续计算准确率
        self.count_rw(pred, label, label.shape[0])

        # 每50个批次记录损失并计算准确率
        if batch_idx%50==0:
            key_list = list(losses.keys())
            for i, key in enumerate(key_list):
                tb.add_scalar('running/' + key, losses[key], self.global_step)

            acc = 100 * self.right_count / (self.right_count + self.wrong_count)
            self.log('running/acc', acc)

        return losses

    def denseCRF(self, cam, label, img_denorm_rgb_hwc):
        # CRF 参数
        alpha = 0.5  # 背景得分的幂运算参数
        t_crf = 10  # CRF 推理的迭代次数
        scale_factor = 1  # 缩放因子
        num_classes = 2  # 类别数量
        # cam_shape: [4,2,256,256]
        B, _, H, W = cam.shape
        # print("B is {}".format(B))
        cam = F.relu(cam)
        cam = cam * label.view(B, 2, 1, 1).to(self.device)
        cam = self.max_norm(cam) 
        cam_np = cam.cpu().detach().numpy() # (3,2,256,256)
        img_np = img_denorm_rgb_hwc.cpu().numpy() # (3,256,256,3)
        crf_results = []
        for i in range(B):
            sample = cam_np[i] # shape: (2, 256, 256)
            image_sample = img_np[i] # (256, 256, 3)
            img_uint8 = image_sample.astype(np.uint8)
            if (self.current_epoch + 1) % 10 ==0 :
                plt.imsave("label_out/img_denorm_%s.png"%i, img_uint8)

            cam_cls1 = sample[1] # 取出类别 1 的 CAM，shape: (256, 256)
            v = cam_cls1.reshape(1, 65536)
            # 计算背景得分
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            # 合并背景得分和CAM值
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = crf_inference_ysh(img_uint8, bgcam_score, t=t_crf, scale_factor=scale_factor, labels=num_classes)
            # 获取预测结果（取最大概率的类别）
            predict = crf_score.argmax(axis=0).astype(np.uint8)  # shape: (H, W)
            if (self.current_epoch + 1) % 10 ==0 : 
                plt.imsave("label_out/label_%s.png"%i, predict, cmap='gray')
            # 转为 tensor 并添加 batch 和 channel 维度
            predict_tensor = torch.from_numpy(predict).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
            crf_results.append(predict_tensor)
        # 拼接成 [B, H, W]
        cam_label = torch.cat(crf_results, dim=0).to(self.device).float() # (3,256,256)
        return cam_label



    def validation_step(self, batch, batch_idx):

        tb = self.logger.experiment

        name = batch[0][0]
        imgs = batch[1]
        label = batch[2]
        
        # 使用单尺度输入（使用第2个尺度, 即原图像尺寸）
        img_ori = imgs[2]  # torch.Size([1, 3, 256, 256])
        B, _, H, W = img_ori.shape
        # print(img_ori.shape)

        with torch.no_grad():
            self.net_recon.eval()
            self.seg_decoder.eval()

            _, _, _, feat_out = self.net_recon(img_ori)
            _, _, _, feat_out_main = self.net_main(img_ori)
            

            seg_output = self.seg_decoder(feat_out, feat_out_main)

            seg_output = seg_output.data.cpu().numpy().squeeze(0).squeeze(0)
            print_true = False
            if print_true:
                print(seg_output)
                print(np.max(seg_output))
                print(seg_output.shape)
            seg_mask = (seg_output >= 0.5).astype(np.int8)
            seg_mask =  seg_mask * 255
            
            if print_true:
                print(seg_mask)
                print(np.max(seg_mask))
                print(seg_mask.shape)
                exit()

        # print(seg_mask.shape) # (256,256)

        ## Export
        # Save segment results
        seg_save_path = os.path.join(tb.log_dir+'/seg_mask/')
        if not os.path.exists(seg_save_path):
          os.makedirs(seg_save_path) 
        cv2.imwrite(os.path.join(seg_save_path, name + '.png'), seg_mask)
        
        
        

    def training_epoch_end(self, training_step_outputs):       
        
        phase = 'train'
        step_outputs = training_step_outputs

        key_list = list(step_outputs[0].keys())
        if 'loss' in key_list:
            key_list.remove('loss')
        loss_list = [0]*len(key_list)

        for output in step_outputs:
            for i, key in enumerate(key_list):
                loss_list[i] += output[key]

        for i, key in enumerate(key_list):
            self.log(phase + '/' + key, loss_list[i] / len(step_outputs))
        
        acc = 100 * self.right_count / (self.right_count + self.wrong_count)
        self.log(phase + '/acc', acc)


    def validation_epoch_end(self, val_step_outputs):

        tb = self.logger.experiment
        phase = 'val'

        if self.check_sanity:
            self.check_sanity = False
            return

        metric_dict = eval_seg(eval_list='infer_226_name', pred_dir=tb.log_dir+'/seg_mask')

        th = metric_dict['th']
        miou = metric_dict['miou']
        mp = metric_dict['mp']
        mr = metric_dict['mr']
        IoU_0_temp = metric_dict['IoU_0']
        IoU_1_temp = metric_dict['IoU_1']
        

        print('Epo ' + str(self.current_epoch).zfill(2) + 
            ' : miou=' + str(round(miou,2)) + 
            ', mP=' + str(round(mp,2)) +
            ', mR=' + str(round(mr,2)) +
            ', IoU_0=' + str(round(IoU_0_temp,3)) +
            ', IoU_1=' + str(round(IoU_1_temp,3)) +
            'at th ' + str(round(th,2))
            )

        if self.max_miou < miou:
            self.max_miou = miou
            print('New record!')

        self.log('val_miou', miou, self.current_epoch)
        tb.add_scalar('metrics/mIoU', miou, self.current_epoch)
        tb.add_scalar('metrics/mP', mp, self.current_epoch)
        tb.add_scalar('metrics/mR', mr, self.current_epoch)
        tb.add_scalar('metrics/th', th, self.current_epoch)


    ## INFER-RELATED
    
    def load_pretrained(self, load_path):
        self.load_pl_dict(self.net_main, load_path, 'net_main')
    
    def load_pretrained_seg(self, load_path):
        self.load_pl_dict(self.net_main, load_path, 'net_main')
        self.load_pl_dict(self.net_recon, load_path, 'net_recon')
        self.load_pl_dict(self.seg_decoder, load_path, 'seg_decoder')

    def seg_test_recon_0416(self, image_tensor):
        with torch.no_grad():
            self.net_recon.eval()
            self.seg_decoder.eval()
            self.net_main.eval()

            _, _, _, feat_out = self.net_recon(image_tensor)
            _, _, _, feat_out_main = self.net_main(image_tensor)
            seg_output = self.seg_decoder(feat_out, feat_out_main)

            # seg_output = self.seg_decoder(feat_out)  # [1, 1, 256, 256]
            seg_output = seg_output.data.cpu().numpy().squeeze(0).squeeze(0)
            print_true = False
            if print_true:
                print(seg_output)
                print(np.max(seg_output))
                print(seg_output.shape)
            seg_mask = (seg_output >= 0.5).astype(np.int8)
            # seg_mask = np.asarray(np.argmax(seg_output, axis=0), dtype=np.uint8)
            if print_true:
                print(seg_mask)
                print(np.max(seg_mask))
                print(seg_mask.shape)
                exit()

        return seg_mask

    def infer(self, batch, vis_path=None, cam_path=None, crf_path=None, alphas=[4,24]):

        # print(batch[0]): ['111212-1']
        # print(batch[2]): tensor([[0., 1.]])
        # num_scales = len(imgs)//2  8/2=4 

        name = batch[0][0]
        imgs = batch[1]
        label = batch[2]

        # for i, img in enumerate(imgs):
        #     print(f"Scale {i}: {img.shape}")
        #Scale 0: torch.Size([1, 3, 128, 128])
        # Scale 1: torch.Size([1, 3, 128, 128])
        # Scale 2: torch.Size([1, 3, 256, 256])
        # Scale 3: torch.Size([1, 3, 256, 256])
        # Scale 4: torch.Size([1, 3, 384, 384])
        # Scale 5: torch.Size([1, 3, 384, 384])
        # Scale 6: torch.Size([1, 3, 512, 512])
        # Scale 7: torch.Size([1, 3, 512, 512])
        # exit()

        # 使用单尺度输入（使用第2个尺度, 即原图像尺寸）
        img_ori = imgs[2]
        B, _, H, W = img_ori.shape

        with torch.no_grad():
            gt = label[0].cpu().numpy() # [0. 1.]
            gt_cls = np.nonzero(gt)[0]  # [1]

            print(gt)
            print(gt_cls)

            self.net_main.eval()

            # 初始化CAM
            cam = torch.zeros(1, 2, H, W, device=self.device)

            # 直接使用单尺度输入进行推理
            img_temp = img_ori.to(self.device)
            cam_temp, _, _, _ = self.net_main(img_temp)

            # print(cam_temp.shape)  torch.Size([1, 2, 32, 32])

            # 插值到原始图像大小
            cam_temp = F.interpolate(cam_temp, (H, W), mode='bilinear', align_corners=False)
            cam_temp = F.relu(cam_temp)    # F.relu 的应用通常是为了确保输出的非负性和数值稳定性

            cam += cam_temp

            cam = cam * label.view(1, 2, 1, 1).to(self.device)
            cam = self.max_norm(cam)   # 进行归一化处理，以确保其在 [0, 1] 范围内
            cam_np = cam.cpu().numpy()[0]  # C x H x W
            
            cam_dict = {}
            for i in range(2):
                # label[0, 0]: tensor(0.)
                # label[0, 1]: tensor(1.)
                if label[0, i] > 1e-5:
                    cam_dict[i] = cam_np[i]

            # cam_dict = {i: cam_np[i] for i in range(2) if label[0, i] > 1e-5}  

            exit()

            if vis_path is not None:
                img_np = denorm(img_ori[0]).cpu().numpy().transpose(1, 2, 0)
                for c in gt_cls:
                    save_img(vis_path + '/' + name + '_cam_' + self.categories[c] + '.png', img_np, cam_np[c])

            if cam_path is not None:
                np.save(cam_path + '/' + name + '.npy', cam_dict)

            if crf_path is not None:
                for a in alphas:
                    crf_dict = _crf_with_alpha_deepcrack(cam_dict, name, alpha=a)
                    np.save(crf_path + '/' + str(a).zfill(2) + '/' + name + '.npy', crf_dict)

    ################ Functions #################   
    def CE_loss(self, predictions, targets):
    # """
    # 计算交叉熵损失
    # :param predictions: 模型的预测输出，形状为 (N, C, H, W)，其中 N 是批量大小，C 是类别数
    # :param targets: 真实标签，形状为 (N, H, W)
    # :return: 交叉熵损失
    # """
        return F.cross_entropy(predictions, targets)


    def max_norm(self, cam):
        N, C, H, W = cam.size()
        cam = F.relu(cam)
        max_v = torch.max(cam.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam = F.relu(cam - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam

    def masked_L1(self, target, img, mask, valid=None):
        if valid==None:
            return self.L1(target*mask, img*mask)
        else:
            if valid.sum()==0:
                return 0
            else:
                # valid: 仅对被标记为有效的样本计算损失
                return self.L1((target*mask)[valid], (img*mask)[valid])

    def clip(self, img):
        img[img<0] = 0
        img[img>1] = 1
        return img

    def split_label(self, label, B):
        label_mask= torch.zeros_like(label)
        # in my experiment, bs=4, that's B=4
        for i in range(B):
            label_idx = torch.nonzero(label[i], as_tuple=False)
            # label: tensor([[0., 1.], [1., 0.], [0., 1.], [0., 1.]]
            # label_idx: tensor([[1]]), tensor([[0]]), tensor([[1]]), tensor([[1]])
            # 生成一个随机整数 rand_idx，该值从 0 到 len(label_idx) 范围内随机选择，表示从 label_idx 中选择一个位置。
            # len(label_idx) = 1, so rand_idx will be 0
            rand_idx = torch.randint(0, len(label_idx), (1,))
            # 使用 rand_idx 从 label_idx 中选择一个索引 target，这就是选中的标签位置。
            target = label_idx[rand_idx][0]
            label_mask[i, target] = 1
        label_remain = label - label_mask
        # print(label)
        # print(label_mask)
        # print(label_remain)
        return label_mask, label_remain

    def count_rw(self, pred, label, bs):
        for b in range(bs):
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            num = len(gt)
            pred_t = pred[b].cpu().detach().numpy()
            pred_cls = pred_t.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count += 1
                else:
                    self.wrong_count += 1

    def load_pl_dict(self, net, pl_dict_path, name):
        pl_dict= torch.load(pl_dict_path)['state_dict']
        len_name = len(name)
        temp_dict = {}
        for key in pl_dict.keys():
            if key[:len_name]==name:
                key_name = key[len_name+1:] # Remove "xxx." at the first of each key
                temp_dict[key_name] = pl_dict[key]
        net.load_state_dict(temp_dict, strict=True)

    def normalize_T(self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Assume that img is in the range of [-1,1]
        img = (img+1)/2. # Now in [0,1]
        for i in range(3):
            img[:,i,:,:] = img[:,i,:,:] - mean[i]
            img[:,i,:,:] = img[:,i,:,:]/std[i]
        
        return img 
    ############################################

    def configure_optimizers(self):
        return None