import torch
import torch.nn as nn
import torch.nn.functional as F

# 增强局部对比度，让目标区域更突出
class ContrastEnhancer(torch.nn.Module):
    def __init__(self, beta=1.0):
        """
        beta: 对比增强强度，越大越激进
        """
        super(ContrastEnhancer, self).__init__()
        self.beta = beta

    def forward(self, feat, cam):
        """
        feat: 特征图 (B, C, H, W) — 来自 backbone
        cam:  激活图 (B, num_classes, H, W) — CAM输出

        返回增强后的 cam
        """

        B, _, H, W = cam.shape
        _, _, h, w = feat.shape # ← 添加这一行，提取 h 和 w

        # 计算局部平均特征（平滑）
        feat_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)

        # 差异表示局部对比
        contrast = (feat - feat_mean).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)

        # 归一化 contrast 到 [0,1]
        contrast = (contrast - contrast.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]) / \
                   (contrast.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] - 
                    contrast.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0] + 1e-5)

        # 🔥 这里新增一行：把 contrast 上采样到 cam 的大小
        if (h != H) or (w != W):
            contrast = F.interpolate(contrast, size=(H, W), mode='bilinear', align_corners=False)

        # 将 contrast 作为加权，增强 cam
        cam = cam * (1 + self.beta * contrast)

        # 再归一化到 [0,1]
        cam = (cam - cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]) / \
              (cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] - 
               cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0] + 1e-5)

        return cam

# 惩罚碎片化，鼓励连通
class ConnectivityLoss(nn.Module):
    def __init__(self, threshold=0.5, use_soft_mask=True, focus_on_target=False, smooth_factor=1.0, reduction='mean'):
        """
        threshold: CAM二值化的阈值
        use_soft_mask: 是否使用soft sigmoid代替硬阈值
        focus_on_target: 是否只对目标区域计算连通性损失（segment_t区域）
        """
        super(ConnectivityLoss, self).__init__()
        self.threshold = threshold
        self.use_soft_mask = use_soft_mask
        self.focus_on_target = focus_on_target
        self.smooth_factor = smooth_factor
        self.reduction = reduction

    def forward(self, cam, segment_t=None):
        """
        cam: (B, 1, H, W)
        segment_t: (B, 1, H, W) 伪标签目标区域（可选）
        """
        B, C, H, W = cam.shape

        # cam最大化类别维度，得到最终mask (假设每个位置只关心最激活的类)
        cam_max, _ = cam.max(dim=1, keepdim=True)  # (B,1,H,W)

        # soft mask or hard threshold
        if self.use_soft_mask:
            # 使用sigmoid代替硬阈值，平滑梯度
            binary_mask = torch.sigmoid((cam_max - self.threshold) * 2)
        else:
            # 硬阈值（不可导）
            binary_mask = (cam_max > self.threshold).float()

        # # 二值化
        # binary_mask = (cam_max > self.threshold).float()

        # 如果设置了只在目标区域计算
        if self.focus_on_target and segment_t is not None:
            # 避免 segment_t 为整数类型
            segment_t = segment_t.float()
            binary_mask = binary_mask * segment_t  # 只保留目标区域

        # 计算每个像素与周围像素的一致性（局部连通性）
        # 加权邻域差异
        loss = 0.0
        weights = {(0,1):1.0, (1,0):1.0, (0,-1):1.0, (-1,0):1.0,
                    (1,1):0.7, (-1,1):0.7, (1,-1):0.7, (-1,-1):0.7}
        for (dy, dx), w in weights.items():
            shifted = torch.roll(binary_mask, shifts=(dy, dx), dims=(2, 3))
            loss += w * F.l1_loss(binary_mask, shifted, reduction=self.reduction)

        loss = loss / sum(weights.values())

        return loss


# def cam_consistency_loss(cam1, cam2, mask=None):
#     """
#     cam1, cam2: shape [B, C, H, W], normalized CAMs
#     mask: optional, shape [B, 1, H, W], if you only want to compute loss on foreground
#     """
#     loss = torch.abs(cam1 - cam2)
#     if mask is not None:
#         loss = loss * mask
#     return loss.mean()

def cam_consistency_loss(cam1, cam2, mask=None):
    """
    cam1, cam2: shape [B, C, H, W], normalized CAMs
    mask: optional, shape [B, 1, H, W], broadcasting to [B, C, H, W]
    """
    # Basic difference
    loss = torch.abs(cam1 - cam2)

    if mask is not None:
        mask = mask.expand_as(cam1)  # Ensure same shape for broadcast
        valid = mask.sum()
        if valid == 0 or torch.isnan(valid):
            return torch.tensor(0.0, device=cam1.device, requires_grad=True)  # 防止除以0 or 全背景
        return (loss * mask).sum() / (valid + 1e-6)  # 数值安全平均
    else:
        return loss.mean()  # 覆盖全图平均


def edge_consistency_loss(cam1, cam2):
    """
    cam1, cam2: [B, 1, H, W] or [B, C, H, W]
    使用 Sobel 算子计算梯度，然后比较两者的边缘结构
    """
    def gradient(img):
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    grad1 = gradient(cam1)
    grad2 = gradient(cam2)

    return F.l1_loss(grad1, grad2)


def center_enhance(cam, sigma=0.2):
    """
    对 CAM 做中心增强处理，使其更加聚焦于高置信度区域的中心。
    cam: [B, 1, H, W]，单通道 CAM
    sigma: 控制高斯的扩散程度
    """
    B, C, H, W = cam.shape
    device = cam.device

    # 计算加权中心点
    x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, C, H, W)
    y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, C, H, W)

    x_center = (cam * x_coords).sum(dim=(2, 3)) / (cam.sum(dim=(2, 3)) + 1e-6)  # [B, C]
    y_center = (cam * y_coords).sum(dim=(2, 3)) / (cam.sum(dim=(2, 3)) + 1e-6)  # [B, C]
    
    total = cam.sum(dim=(2, 3), keepdim=True)
    if (total < 1e-4).any():
        return cam   # 放弃增强，返回原来

    else:
        # 构造高斯权重图
        x_grid = x_coords
        y_grid = y_coords
        x_center = x_center.view(B, C, 1, 1)
        y_center = y_center.view(B, C, 1, 1)

        gauss = torch.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))  # [B, C, H, W]

        # 中心增强：高斯加权
        cam_enhanced = cam * gauss
        return cam_enhanced

def normalize(cam):
    B, C, H, W = cam.shape
    cam_min = cam.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
    cam_max = cam.view(B, C, -1).max(dim=-1)[0].view(B, C, 1, 1)
    return (cam - cam_min) / (cam_max - cam_min + 1e-6)