import torch
import torch.nn as nn
import torch.nn.functional as F



class SobelEdge(nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()
        kernel_x = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x):
        # x: [B, 3, H, W] -> 转为灰度
        x_gray = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        edge_x = F.conv2d(x_gray, self.weight_x, padding=1)
        edge_y = F.conv2d(x_gray, self.weight_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge  # [B, 1, H, W]


class SAARRefiner(nn.Module):
    def __init__(self, edge_guidance=True):
        super(SAARRefiner, self).__init__()
        self.edge_guidance = edge_guidance
        self.affinity_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, groups=1, bias=False)
        self.propagation = nn.Conv2d(8, 1, kernel_size=1)
        if self.edge_guidance:
            self.edge_detector = SobelEdge()

    def forward(self, cam, image):
        """
        cam: [B, 2, H, W] - class activation maps
        image: [B, 3, H, W] - denormalized RGB image
        return: [B, 1, H, W] - pseudo label mask
        """
        cam = F.relu(cam)
        cam = cam / (cam.sum(dim=1, keepdim=True) + 1e-6)  # normalize across class dim
        cam_cls1 = cam[:, 1:2, :, :]  # [B, 1, H, W] - foreground class

        # Step 1: Compute directional affinity
        affinity = self.affinity_conv(cam_cls1)  # [B, 8, H, W]

        # Step 2: Optional edge guidance
        if self.edge_guidance:
            edge_map = self.edge_detector(image)  # [B, 1, H, W]
            affinity = affinity * torch.sigmoid(edge_map)

        # Step 3: Propagate + refine
        refined = self.propagation(affinity)  # [B, 1, H, W]
        refined = torch.sigmoid(refined)

        # Step 4: Binarize (or threshold)
        pseudo_mask = (refined > 0.5).float()  # or keep float if using as soft label

     #    pseudo_mask_png = pseudo_mask.astype(np.int8)
     #    pseudo_mask_png = pseudo_mask_png * 255
     #    print()

        return pseudo_mask