import torch
from torch import nn
import numpy as np
import pdb
# from torch.nn.common_types import T

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False, start_imd = False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False,start_imd=False):
        if not start_imd:
            branch2 = self.bn_branch2a(x)
            branch2 = F.relu(branch2)
        else:
            branch2 = x
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False, start_imd=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu,start_imd=start_imd)

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)




class ReconBlock(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            ks = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            *args, **kwargs):
        super(ReconBlock, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding= padding,
                dilation = dilation,
                bias = False)
        self.bn1 = nn.InstanceNorm2d(out_chan)
        self.gelu = nn.GELU()    

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_chan))
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)

        if self.downsample == None:
            inten = x
        else:
            inten = self.downsample(x)
        out = residual + inten
        out = self.gelu(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)






class Net_main(nn.Module):
    def __init__(self,D):
        super(Net_main, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 256, 1, bias=False)
        # self.fc9 = nn.Conv2d(256, 20, 1, bias=False)
        self.fc9 = nn.Conv2d(256, 2, 1, bias=False)

        self.fc_recon = nn.Conv2d(4096,D,1,bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.5)
        
        
        self.dr1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.dr2 = torch.nn.Conv2d(1024, 128, 1, bias=False)

        self.genUnet = GeneratorUNet(in_channels=(64+128+D),out_channels=3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = []#self.conv1a, self.b2, self.b2_1, self.b2_2]
    
    def recon_decoder(self, feat, use_tanh=False):
        recon_img = self.genUnet(feat)
        if use_tanh:
            return F.tanh(recon_img)
        else:
            return recon_img

    def forward(self, img):

        _, _, H, W = img.shape

        x = self.conv1a(img)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56

        feat_out = x

        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x,get_x_bn_relu=True)  # B x 2048 x 56 x 56

        x = self.b7(x)
        feat = F.relu(self.bn7(x))

        x = F.relu(self.fc8(feat))

        cam = self.fc9(x)

        conv4 = self.dr1(conv4)
        conv5 = self.dr2(conv5)
        feat = self.fc_recon(feat)

        feat_final = torch.cat([conv4,conv5,feat],dim=1)


        out = self.avgpool(cam).squeeze(3).squeeze(2)

        return cam, out, feat_final, feat_out

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m,nn.ConvTranspose2d)):# or isinstance(m,nn.modules.instancenorm.InstanceNorm2d)): #
                if m.weight.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net_main_bd(nn.Module):
    def __init__(self,D):
        super(Net_main_bd, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 256, 1, bias=False)
        # self.fc9 = nn.Conv2d(256, 20, 1, bias=False)
        self.fc9 = nn.Conv2d(256, 2, 1, bias=False)

        self.fc_recon = nn.Conv2d(4096,D,1,bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.5)
        
        
        self.dr1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.dr2 = torch.nn.Conv2d(1024, 128, 1, bias=False)

        self.genUnet = GeneratorUNet(in_channels=(64+128+D),out_channels=3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = []#self.conv1a, self.b2, self.b2_1, self.b2_2]

        # add by nachuan:
        # ---------- 新增部分：边界预测头 ----------
        self.boundary_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0~1的概率图
        )

    
    def recon_decoder(self, feat, use_tanh=False):
        recon_img = self.genUnet(feat)
        if use_tanh:
            return F.tanh(recon_img)
        else:
            return recon_img

    def forward(self, img):

        _, _, H, W = img.shape

        x = self.conv1a(img)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        

        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56

        feat_out = x

        # ---------- 新增部分：生成边界预测 ----------
        # conv3.shape: torch.Size([1, 256, 64, 64])
        # conv3.shape: torch.Size([1, 512, 56, 56])
        boundary_pred = self.boundary_head(conv4)  # Bx1xH/8xW/8
        boundary_pred = F.interpolate(boundary_pred, size=(H, W), mode='bilinear')  # 上采样到原图分辨率
        # ---------- 新增部分结束 ----------

        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x,get_x_bn_relu=True)  # B x 2048 x 56 x 56

        x = self.b7(x)
        feat = F.relu(self.bn7(x))

        x = F.relu(self.fc8(feat))

        cam = self.fc9(x)

        conv4 = self.dr1(conv4)
        conv5 = self.dr2(conv5)
        feat = self.fc_recon(feat)

        feat_final = torch.cat([conv4,conv5,feat],dim=1)

        out = self.avgpool(cam).squeeze(3).squeeze(2)

        return cam, out, feat_final, feat_out, boundary_pred  # 增加边界预测输出

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m,nn.ConvTranspose2d)):# or isinstance(m,nn.modules.instancenorm.InstanceNorm2d)): #
                if m.weight.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


# ---------- 新增部分：多尺度边界预测头 ----------
class MultiScaleBoundaryHead(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBoundaryHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),  # 适配输入通道数
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0~1的概率图
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Net_main_bd_multi(nn.Module):
    def __init__(self,D):
        super(Net_main_bd_multi, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 256, 1, bias=False)
        # self.fc9 = nn.Conv2d(256, 20, 1, bias=False)
        self.fc9 = nn.Conv2d(256, 2, 1, bias=False)

        self.fc_recon = nn.Conv2d(4096,D,1,bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.5)
        
        
        self.dr1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.dr2 = torch.nn.Conv2d(1024, 128, 1, bias=False)

        self.genUnet = GeneratorUNet(in_channels=(64+128+D),out_channels=3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = []#self.conv1a, self.b2, self.b2_1, self.b2_2]

        # add by nachuan:
        # ---------- 新增部分：边界预测头 ----------
        # self.boundary_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 1, kernel_size=1),
        #     nn.Sigmoid()  # 输出0~1的概率图
        # )
        # self.boundary_head = MultiScaleBoundaryHead()

    
    def recon_decoder(self, feat, use_tanh=False):
        recon_img = self.genUnet(feat)
        if use_tanh:
            return F.tanh(recon_img)
        else:
            return recon_img

    def forward(self, img):

        _, _, H, W = img.shape

        x = self.conv1a(img)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        conv2 = x # 显式保存 conv2


        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        

        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x,get_x_bn_relu=True)  # B x 2048 x 56 x 56

        x = self.b7(x)
        feat = F.relu(self.bn7(x))

        x = F.relu(self.fc8(feat))

        cam = self.fc9(x)

        conv4 = self.dr1(conv4)
        conv5 = self.dr2(conv5)
        feat = self.fc_recon(feat)

        feat_final = torch.cat([conv4,conv5,feat],dim=1)

        out = self.avgpool(cam).squeeze(3).squeeze(2)

        # # ---------- 新增部分：生成边界预测 ----------
        # # conv3.shape: torch.Size([1, 256, 64, 64])
        # boundary_pred = self.boundary_head(conv3)  # Bx1xH/8xW/8
        # boundary_pred = F.interpolate(boundary_pred, size=(H, W), mode='bilinear')  # 上采样到原图分辨率
        # # ---------- 新增部分结束 ----------

        # ---------- 新增部分：多尺度边界预测 ----------
        # 获取不同尺度的通道数
        conv2_channels = conv2.shape[1]  # 128
        conv3_channels = conv3.shape[1]  # 256
        conv4_channels = conv4.shape[1]  # 512

        device = x.device

        # 适配不同通道数
        boundary_head_2 = MultiScaleBoundaryHead(conv2_channels).to(device)
        boundary_head_3 = MultiScaleBoundaryHead(conv3_channels).to(device)
        boundary_head_4 = MultiScaleBoundaryHead(conv4_channels).to(device)

        # 计算不同尺度的边界预测
        boundary_pred_2 = boundary_head_2(conv2)  # 低级特征
        boundary_pred_3 = boundary_head_3(conv3)  # 中级特征
        boundary_pred_4 = boundary_head_4(conv4)  # 高级特征

        # 上采样到原图分辨率
        boundary_pred_2 = F.interpolate(boundary_pred_2, size=(H, W), mode='bilinear', align_corners=False)
        boundary_pred_3 = F.interpolate(boundary_pred_3, size=(H, W), mode='bilinear', align_corners=False)
        boundary_pred_4 = F.interpolate(boundary_pred_4, size=(H, W), mode='bilinear', align_corners=False)

        boundary_pred = [boundary_pred_2, boundary_pred_3, boundary_pred_4]

        return cam, out, feat_final, boundary_pred  # 增加边界预测输出

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m,nn.ConvTranspose2d)):# or isinstance(m,nn.modules.instancenorm.InstanceNorm2d)): #
                if m.weight.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net_recon(nn.Module):
    def __init__(self,D):
        super(Net_recon, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 256, 1, bias=False)
        # self.fc9 = nn.Conv2d(256, 20, 1, bias=False)
        self.fc9 = nn.Conv2d(256, 2, 1, bias=False)

        self.fc_recon = nn.Conv2d(4096,D,1,bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.5)
        
        
        self.dr1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.dr2 = torch.nn.Conv2d(1024, 128, 1, bias=False)

        self.genUnet = GeneratorUNet(in_channels=(64+128+D),out_channels=3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = []#self.conv1a, self.b2, self.b2_1, self.b2_2]
    
    def recon_decoder(self, feat, use_tanh=False):
        recon_img = self.genUnet(feat)
        if use_tanh:
            return F.tanh(recon_img)
        else:
            return recon_img

    def forward(self, img):

        _, _, H, W = img.shape

        x = self.conv1a(img)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 512 x 56 x 56

        # feat_out = x

        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56
        
        feat_out = x

        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x,get_x_bn_relu=True)  # B x 2048 x 56 x 56
        

        x = self.b7(x)
        feat = F.relu(self.bn7(x))

        x = F.relu(self.fc8(feat))

        cam = self.fc9(x)

        conv4 = self.dr1(conv4)
        conv5 = self.dr2(conv5)
        feat = self.fc_recon(feat)

        feat_final = torch.cat([conv4,conv5,feat],dim=1)

        

        out = self.avgpool(cam).squeeze(3).squeeze(2)

        return cam, out, feat_final, feat_out

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m,nn.ConvTranspose2d)):# or isinstance(m,nn.modules.instancenorm.InstanceNorm2d)): #
                if m.weight.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if isinstance(m,nn.ConvTranspose2d):
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups




##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size,affine=False))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size,affine=False),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)


        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, out_channels, 3, padding=1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)


        return self.final(u7)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class EdgeDetector(nn.Module):
    def __init__(self):
        super(EdgeDetector, self).__init__()

        # 第一层卷积：输入3通道（RGB），输出8通道
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二层卷积：输入8通道，输出16通道
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        # 第三层卷积：输入16通道，输出1通道（边缘概率图）
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # 输出范围[0,1]

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        """
        输入: img (B x 3 x H x W)
        输出: edge_prob (B x 1 x H x W)，表示边缘概率图
        """
        # 第一层卷积
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 第三层卷积
        x = self.conv3(x)
        edge_prob = self.sigmoid(x)  # 输出边缘概率图

        return edge_prob
    
    def train(self, mode=True):
        """
        训练模式设置
        """
        super().train(mode)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # BN层固定
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
        return

    def get_parameter_groups(self):
        """
        获取参数分组（用于优化器）
        返回格式: [weights, biases]
        """
        groups = ([], [])  # weights, biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    groups[0].append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    groups[1].append(m.bias)
        return groups



class SimpleSegDecoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleSegDecoder, self).__init__()
        
         # 定义卷积层
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        # 定义反卷积层（上采样）
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        # 前向传播 x: [4, 896, 32, 32]
        # 前向传播 x: [4, 448, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))  # → [4, 512, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)))  # → [4, 256, 32, 32]
        x = F.relu(self.bn3(self.conv3(x)))  # → [4, 128, 32, 32]
        
        x = F.relu(self.bn4(self.deconv1(x)))  # → [4, 64, 64, 64]
        x = F.relu(self.bn5(self.deconv2(x)))  # → [4, 32, 128, 128]
        x = self.deconv3(x)                    # → [4, 1, 256, 256]
        
        logits = torch.sigmoid(x)
        return logits
    
    def get_parameter_groups(self):
        # 将参数分为两组：权重和偏置/批归一化
        weight_params = []
        bias_bn_params = []
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                weight_params.append(param)
            else:
                bias_bn_params.append(param)
        
        return [
            {'params': weight_params, 'lr': 1},  # 权重参数
            {'params': bias_bn_params, 'lr': 2}  # 偏置和批归一化参数
        ]


class SEBlock_fusion1(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock_fusion1, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels),
                    nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)  # (B, N, out_dim)
        B, N, out_dim = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, out_dim)
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)  # (B, N, N)

        # print(type(adj))  # Debugging
        # print(adj.shape)  # Ensure it's a tensor of shape (B, N, N) or (B, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(type(attention))  # Ensure attention is a tensor

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # (B, N, out_dim)
        return F.elu(h_prime)


class PathGraphModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PathGraphModule, self).__init__()
        self.gat1 = GraphAttentionLayer(in_ch, out_ch)
        self.gat2 = GraphAttentionLayer(out_ch, out_ch)

    def forward(self, x, adj):
        B, C, H, W = x.size()
        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x = self.gat1(x, adj)
        x = self.gat2(x, adj)
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        return x


def build_4neighbor_adj(H, W, B, device):
    N = H * W
    adj = torch.zeros(B, N, N, device=device)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * W + j)
            if i < H - 1:
                neighbors.append((i + 1) * W + j)
            if j > 0:
                neighbors.append(i * W + j - 1)
            if j < W - 1:
                neighbors.append(i * W + j + 1)
            for n in neighbors:
                adj[:, idx, n] = 1
    return adj

class G_PFFM(nn.Module):
    def __init__(self, in_channels, reduced_channels=128):
        super(G_PFFM, self).__init__()
        self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.graph = PathGraphModule(reduced_channels, reduced_channels)
        self.se = SEBlock(reduced_channels)
        self.fuse = nn.Conv2d(reduced_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels // 2)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)  # (B, C1+C2, H, W)
        x = self.reduce(x)
        B, C, H, W = x.size()
        adj = build_4neighbor_adj(H, W, B, x.device)
        x = self.graph(x, adj)
        x = self.se(x)
        x = F.relu(self.bn(self.fuse(x)))
        return x

class SimpleSegDecoder_fusion_GPFFM(nn.Module):
    def __init__(self, input_channels1, input_channels2, num_classes, reduction=16):
        super(SimpleSegDecoder_fusion_GPFFM, self).__init__()
        
        total_input_channels = input_channels1 + input_channels2

        # # 1x1卷积融合两个输入特征
        # self.fusion_conv = nn.Conv2d(total_input_channels, total_input_channels // 2, kernel_size=1)
        # self.bn_fusion = nn.BatchNorm2d(total_input_channels // 2)
        # # SE注意力模块
        # self.se_block = SEBlock_fusion1(total_input_channels // 2, reduction)

        # 用路径感知融合模块替换之前的融合与SE部分
        self.path_fusion = G_PFFM(in_channels=total_input_channels, reduced_channels= 128)

        # 定义卷积层
        self.conv1 = nn.Conv2d(total_input_channels // 2, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        # 定义反卷积层（上采样）
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat1, feat2):
        
        # # 拼接两个输入特征
        # x = torch.cat([feat1, feat2], dim=1)  # [B, C1+C2, H, W]
        # # 融合卷积 + BN + ReLU
        # x = F.relu(self.bn_fusion(self.fusion_conv(x))) 
        # # 通道注意力（SE Block）
        # x = self.se_block(x)

        # 路径感知特征融合
        x = self.path_fusion(feat1, feat2)  # [B, (C1+C2)//2, H, W]

        # 前向传播 x: [4, 896, 32, 32]
        # 前向传播 x: [4, 448, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))  # → [4, 512, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)))  # → [4, 256, 32, 32]
        x = F.relu(self.bn3(self.conv3(x)))  # → [4, 128, 32, 32]
        
        x = F.relu(self.bn4(self.deconv1(x)))  # → [4, 64, 64, 64]
        x = F.relu(self.bn5(self.deconv2(x)))  # → [4, 32, 128, 128]
        x = self.deconv3(x)                    # → [4, 1, 256, 256]
        
        logits = torch.sigmoid(x)
        return logits

class SimpleSegDecoder_fusion(nn.Module):
    def __init__(self, input_channels1, input_channels2, num_classes, reduction=16):
        super(SimpleSegDecoder_fusion, self).__init__()
        
        total_input_channels = input_channels1 + input_channels2

        # 1x1卷积融合两个输入特征
        self.fusion_conv = nn.Conv2d(total_input_channels, total_input_channels // 2, kernel_size=1)
        self.bn_fusion = nn.BatchNorm2d(total_input_channels // 2)
        # SE注意力模块
        self.se_block = SEBlock_fusion1(total_input_channels // 2, reduction)
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(total_input_channels // 2, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        # 定义反卷积层（上采样）
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat1, feat2):
        
        # 拼接两个输入特征
        x = torch.cat([feat1, feat2], dim=1)  # [B, C1+C2, H, W]
        # 融合卷积 + BN + ReLU
        x = F.relu(self.bn_fusion(self.fusion_conv(x))) 
        # 通道注意力（SE Block）
        x = self.se_block(x)

        # # 路径感知特征融合
        # x = self.path_fusion(feat1, feat2)  # [B, (C1+C2)//2, H, W]

        # 前向传播 x: [4, 896, 32, 32]
        # 前向传播 x: [4, 448, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))  # → [4, 512, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)))  # → [4, 256, 32, 32]
        x = F.relu(self.bn3(self.conv3(x)))  # → [4, 128, 32, 32]
        
        x = F.relu(self.bn4(self.deconv1(x)))  # → [4, 64, 64, 64]
        x = F.relu(self.bn5(self.deconv2(x)))  # → [4, 32, 128, 128]
        x = self.deconv3(x)                    # → [4, 1, 256, 256]
        
        logits = torch.sigmoid(x)
        return logits
    
    # def get_parameter_groups(self):
    #     # 将参数分为两组：权重和偏置/批归一化
    #     weight_params = []
    #     bias_bn_params = []
        
    #     for name, param in self.named_parameters():
    #         if 'weight' in name:
    #             weight_params.append(param)
    #         else:
    #             bias_bn_params.append(param)
        
    #     return [
    #         {'params': weight_params, 'lr': 1},  # 权重参数
    #         {'params': bias_bn_params, 'lr': 2}  # 偏置和批归一化参数
    #     ]