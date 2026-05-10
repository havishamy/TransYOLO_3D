# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


def visualize_attention_maps(
    attns, batch_idx=0, img_name="test", save_dir=Path("/home/dsj/attention_maps"), resize_to=None, normalize=True
):
    """可视化attns字典中的p2/p3/p4/p5注意力张量.

    Args:
        attns: 模型forward输出的attns字典，{'p2':(B,att_ch,H,W), 'p3':..., 'p4':..., 'p5':...}
        batch_idx: 批量中要可视化的图片索引（B维度），默认第0张
        img_name: 原始图片名（用于命名保存的注意力图），默认test
        save_dir: 注意力图固定保存路径，默认和你边缘图同目录的attention_maps
        resize_to: 可选，将所有尺度注意力图上采样到该尺寸（如self.out_size），方便对比
        normalize: 是否将注意力值归一化到[0,255]，必须为True（否则无法可视化）.
    """
    # 确保保存路径存在
    save_dir.mkdir(parents=True, exist_ok=True)

    # 遍历每个尺度的注意力张量
    for scale, att_feat in attns.items():
        # 提取单个样本的注意力特征：(att_ch, H, W)
        att_single = att_feat[batch_idx].detach().cpu()  # 去掉梯度，转到CPU
        _B, _C, _H, _W = att_feat.shape

        # 核心：多通道聚合为单通道注意力热力图（GAP）：(att_ch, H, W) → (1, H, W)
        att_heatmap = torch.mean(att_single, dim=0, keepdim=True)  # 通道维度求平均

        # 归一化到[0,1]，再转到[0,255]的uint8格式
        if normalize:
            att_min = att_heatmap.min()
            att_max = att_heatmap.max()
            # 防止除0（如果注意力值全相同）
            if att_max - att_min < 1e-6:
                att_heatmap = torch.zeros_like(att_heatmap)
            else:
                att_heatmap = (att_heatmap - att_min) / (att_max - att_min)
        att_heatmap = (att_heatmap * 255).type(torch.uint8).numpy()  # (1, H, W) → np.array

        # 去掉单通道维度，转为OpenCV支持的(H, W)
        att_heatmap = np.squeeze(att_heatmap, axis=0)

        # 可选：上采样到指定尺寸（如原始图像尺寸self.out_size）
        if resize_to is not None:
            att_heatmap = cv2.resize(att_heatmap, dsize=(resize_to[1], resize_to[0]), interpolation=cv2.INTER_LINEAR)

        # 转为3通道灰度图（方便OpenCV保存，和你边缘图格式一致）
        att_heatmap_3ch = cv2.cvtColor(att_heatmap, cv2.COLOR_GRAY2BGR)

        # 生成保存路径：固定路径/尺度_原始名.jpg（如p2_test.jpg）
        save_path = save_dir / f"{scale}_{img_name}.jpg"
        # 保存注意力图
        cv2.imwrite(str(save_path), att_heatmap_3ch)
        print(f"已保存{scale}尺度注意力图：{save_path}")


class SepConv(nn.Module):
    """Depthwise separable conv: DW -> PW, with BN + ReLU."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ASPPLite(nn.Module):
    """A lightweight ASPP-like context module (dilations small)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv_3 = nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3, bias=False)
        self.conv_5 = nn.Conv2d(in_ch, out_ch, 3, padding=5, dilation=5, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.project = nn.Sequential(
            nn.BatchNorm2d(out_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_3(x)
        x3 = self.conv_5(x)
        x4 = self.pool(x)
        x4 = self.pool_conv(x4)
        x4 = F.interpolate(x4, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(out)


class edge_(nn.Module):
    """Lightweight edge/boundary module producing: - per-class edge map at original resolution (B, nc, H, W) - scale
    attentions for P3/P4/P5 (dict of tensors).

    Args:
        in_chs: tuple/list of input channels for (P3, P4, P5) respectively (e.g. (256,512,1024))
        mid_ch: base intermediate channel (controls param count), e.g. 64
        nc: number of classes (21)
        out_size: final full image size (H, W) e.g. (640,640)

    Returns:
        edge_map: (B, nc, H, W)
        attns: dict {'p3': Tensor(B, mid_att_ch, H3, W3), 'p4':..., 'p5':...} these attns are feature maps to be fused
            in neck (you can multiply or concat).
    """

    def __init__(self, in_chs=(128, 256, 512, 1024), mid_ch=64, nc=21, out_size=(160, 160)):
        super().__init__()
        assert len(in_chs) == 4, "in_chs must be (P3,P4,P5)"

        p2_ch, p3_ch, p4_ch, p5_ch = in_chs
        self.nc = nc
        self.out_size = out_size

        # 1x1 reduction to a common channel (lightweight)
        self.reduce_p2 = nn.Sequential(
            nn.Conv2d(p2_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p3 = nn.Sequential(
            nn.Conv2d(p3_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p4 = nn.Sequential(
            nn.Conv2d(p4_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p5 = nn.Sequential(
            nn.Conv2d(p5_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )

        # Upsample p5->p3 resolution, p4->p3 resolution and fuse at p3 scale
        self.aspp = ASPPLite(mid_ch * 4, mid_ch * 2)  # context on fused features

        # refinement convs (lightweight separable convs)
        self.refine = nn.Sequential(
            SepConv(mid_ch * 2, mid_ch * 2),
            SepConv(mid_ch * 2, mid_ch),
        )

        # produce per-class edge map (at p3 resolution then upsample to full)
        self.edge_pred = nn.Conv2d(mid_ch, 1, 1)

        # generate per-scale attention features for neck fusion (1-channel attention per class might be heavy;
        # we produce a compact feature map (mid_ch) for each scale that the BAM can consume)
        att_ch = max(16, mid_ch // 2)
        self.att_p3 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )
        self.att_p4 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )
        self.att_p5 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )

        # small heads to align attention maps to incoming neck channel if you need (optional)
        # self.att_align_p3 = nn.Conv2d(att_ch, p3_ch, 1)  # uncomment if BAM expects same channels

        # lightweight dropout optional
        self.drop = nn.Dropout2d(0.05)

        # initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feats):
        """feats: tuple/list of (p3_feat, p4_feat, p5_feat) where p3 has highest resolution (e.g. 1/8), p5 lowest
        (1/32).

        Returns:
            edge_map_full: (B, nc, H, W) where H,W == self.out_size
            attns: dict of scale features {'p3':..., 'p4':..., 'p5':...}.
        """
        p2, p3, p4, p5 = feats  # expect tensors

        # reduce channels to common mid_ch
        r2 = self.reduce_p2(p2)  # B x mid x H1 x W1
        r3 = self.reduce_p3(p3)  # B x mid x H3 x W3
        r4 = self.reduce_p4(p4)  # B x mid x H4 x W4
        r5 = self.reduce_p5(p5)  # B x mid x H5 x W5

        # upsample r4 and r5 to r3 size
        size2 = r2.shape[2:]
        size3 = r3.shape[2:]
        r3_up = F.interpolate(r3, size=size2, mode="bilinear", align_corners=False)
        r4_up = F.interpolate(r4, size=size2, mode="bilinear", align_corners=False)
        r5_up = F.interpolate(r5, size=size2, mode="bilinear", align_corners=False)

        # fuse at p3 resolution
        fused = torch.cat([r2, r3_up, r4_up, r5_up], dim=1)  # B x (mid*3) x H3 x W3
        fused = self.drop(fused)
        ctx = self.aspp(fused)  # B x (mid*2) x H3 x W3
        feat = self.refine(ctx)  # B x mid x H3 x W3

        # per-class edge prediction at p3 resolution, then upsample to full image
        edge_p2 = self.edge_pred(feat)  # B x nc x H3 x W3
        edge_full = F.interpolate(edge_p2, size=self.out_size, mode="bilinear", align_corners=False)
        edge_map_full = torch.sigmoid(edge_full)  # confidence per class boundary

        # produce per-scale attention features (for BAM/neck)
        feat = F.interpolate(feat, size=size3, mode="bilinear", align_corners=False)
        att_p3 = self.att_p3(feat)  # B x att_ch x H3 x W3
        # downsample feature to p4 and p5 resolution for attention generation
        # compute att on r4/r5 aligned features:
        att_p4 = self.att_p4(r4)  # B x att_ch x H4 x W4
        att_p5 = self.att_p5(r5)  # B x att_ch x H5 x W5

        # Optionally normalize attentions to [0,1] if used multiplicatively:
        # att_p3 = torch.sigmoid(att_p3); att_p4 = torch.sigmoid(att_p4); att_p5 = torch.sigmoid(att_p5)

        attns = {"p3": att_p3, "p4": att_p4, "p5": att_p5}
        return edge_map_full, attns


class edge(nn.Module):
    """Lightweight edge/boundary module producing: - per-class edge map at original resolution (B, nc, H, W) - scale
    attentions for P3/P4/P5 (dict of tensors).

    Args:
        in_chs: tuple/list of input channels for (P3, P4, P5) respectively (e.g. (256,512,1024))
        mid_ch: base intermediate channel (controls param count), e.g. 64
        nc: number of classes (21)
        out_size: final full image size (H, W) e.g. (640,640)

    Returns:
        edge_map: (B, nc, H, W)
        attns: dict {'p3': Tensor(B, mid_att_ch, H3, W3), 'p4':..., 'p5':...} these attns are feature maps to be fused
            in neck (you can multiply or concat).
    """

    def __init__(self, in_chs=(128, 256, 512, 1024), mid_ch=64, nc=21, out_size=(160, 160)):
        super().__init__()
        assert len(in_chs) == 4, "in_chs must be (P3,P4,P5)"

        p2_ch, p3_ch, p4_ch, p5_ch = in_chs
        self.nc = nc
        self.out_size = out_size

        # 1x1 reduction to a common channel (lightweight)
        self.reduce_p2 = nn.Sequential(
            nn.Conv2d(p2_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p3 = nn.Sequential(
            nn.Conv2d(p3_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p4 = nn.Sequential(
            nn.Conv2d(p4_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.reduce_p5 = nn.Sequential(
            nn.Conv2d(p5_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )

        # Upsample p5->p3 resolution, p4->p3 resolution and fuse at p3 scale
        self.aspp = ASPPLite(mid_ch * 4, mid_ch * 2)  # context on fused features

        # refinement convs (lightweight separable convs)
        self.refine = nn.Sequential(
            SepConv(mid_ch * 2, mid_ch * 2),
            SepConv(mid_ch * 2, mid_ch),
        )

        # produce per-class edge map (at p3 resolution then upsample to full)
        self.edge_pred = nn.Conv2d(mid_ch, 1, 1)

        # generate per-scale attention features for neck fusion (1-channel attention per class might be heavy;
        # we produce a compact feature map (mid_ch) for each scale that the BAM can consume)
        att_ch = max(16, mid_ch // 2)
        self.att_p2 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )
        self.att_p3 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )
        self.att_p4 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )
        self.att_p5 = nn.Sequential(
            nn.Conv2d(mid_ch, att_ch, 1, bias=False), nn.BatchNorm2d(att_ch), nn.ReLU(inplace=True)
        )

        # small heads to align attention maps to incoming neck channel if you need (optional)
        # self.att_align_p3 = nn.Conv2d(att_ch, p3_ch, 1)  # uncomment if BAM expects same channels

        # lightweight dropout optional
        self.drop = nn.Dropout2d(0.05)

        # initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feats, img_name="test", vis_batch_idx=0, vis_attention=True):
        """feats: tuple/list of (p3_feat, p4_feat, p5_feat) where p3 has highest resolution (e.g. 1/8), p5 lowest
        (1/32).

        Returns:
            edge_map_full: (B, nc, H, W) where H,W == self.out_size
            attns: dict of scale features {'p3':..., 'p4':..., 'p5':...}.
        """
        p2, p3, p4, p5 = feats  # expect tensors

        # reduce channels to common mid_ch
        r2 = self.reduce_p2(p2)  # B x mid x H1 x W1
        r3 = self.reduce_p3(p3)  # B x mid x H3 x W3
        r4 = self.reduce_p4(p4)  # B x mid x H4 x W4
        r5 = self.reduce_p5(p5)  # B x mid x H5 x W5

        # upsample r4 and r5 to r3 size
        size2 = r2.shape[2:]
        r3_up = F.interpolate(r3, size=size2, mode="bilinear", align_corners=False)
        r4_up = F.interpolate(r4, size=size2, mode="bilinear", align_corners=False)
        r5_up = F.interpolate(r5, size=size2, mode="bilinear", align_corners=False)

        # fuse at p3 resolution
        fused = torch.cat([r2, r3_up, r4_up, r5_up], dim=1)  # B x (mid*3) x H3 x W3
        fused = self.drop(fused)
        ctx = self.aspp(fused)  # B x (mid*2) x H3 x W3
        feat = self.refine(ctx)  # B x mid x H3 x W3

        # per-class edge prediction at p3 resolution, then upsample to full image
        edge_p2 = self.edge_pred(feat)  # B x nc x H3 x W3
        edge_full = F.interpolate(edge_p2, size=self.out_size, mode="bilinear", align_corners=False)
        edge_map_full = torch.sigmoid(edge_full)  # confidence per class boundary

        # produce per-scale attention features (for BAM/neck)
        feat = F.interpolate(feat, size=size2, mode="bilinear", align_corners=False)
        att_p2 = self.att_p2(feat)
        att_p3 = self.att_p3(r3)  # B x att_ch x H3 x W3
        # downsample feature to p4 and p5 resolution for attention generation
        # compute att on r4/r5 aligned features:
        att_p4 = self.att_p4(r4)  # B x att_ch x H4 x W4
        att_p5 = self.att_p5(r5)  # B x att_ch x H5 x W5

        # Optionally normalize attentions to [0,1] if used multiplicatively:
        # att_p3 = torch.sigmoid(att_p3); att_p4 = torch.sigmoid(att_p4); att_p5 = torch.sigmoid(att_p5)

        attns = {"p2": att_p2, "p3": att_p3, "p4": att_p4, "p5": att_p5}
        # -------------------------- 新增：注意力可视化调用 --------------------------
        if vis_attention:
            visualize_attention_maps(
                attns=attns,
                batch_idx=vis_batch_idx,
                img_name=img_name,
                save_dir=Path("/home/dsj/code/yolov5_modify/runs_edge_p2_sim"),  # 注意力图固定保存路径
                resize_to=self.out_size,  # 上采样到原始图像尺寸，方便和边缘图/原图对比
            )
        # -------------------------------------------------------------------------

        return edge_map_full, attns


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=1,
        relu_first=True,
        bias=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        depthwise = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=inplanes,
            bias=bias,
        )
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(nn.ReLU(), depthwise, bn_depth, pointwise, bn_point)
        else:
            self.block = nn.Sequential(
                depthwise, bn_depth, nn.ReLU(inplace=True), pointwise, bn_point, nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class BAM(nn.Module):
    """Boundary Attention Module 输入: - feat: YOLO 主干或 neck 特征 (B, C, H, W) - edge: 边缘特征 (B, 1 or k, H, W).

    输出:
        - 融合后的特征 (B, C, H, W)
    """

    def __init__(self, feat_channels, edge_channels=1, reduction=8):
        super().__init__()

        # Step1: 把 edge 通道压成 1 通道（保持空间信息）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_channels, 1, 3, padding=1),
            nn.Sigmoid(),  # 归一化
        )

        # Step2: 用 1×1 conv 调整 feat 的 channel，增强表达能力
        self.feat_proj = nn.Conv2d(feat_channels, feat_channels, 1)

    def forward(self, x):
        """feat: (B, C, H, W) edge: (B, k, H, W).
        """
        c = x[0]
        att_map = x[1][1]
        sizes = [att_map["p2"].size()[2:], att_map["p3"].size()[2:], att_map["p4"].size()[2:], att_map["p5"].size()[2:]]
        for i in range(len(sizes)):
            if c.size()[2:] == sizes[i]:
                att_map = att_map["p" + str(i + 2)]
                break
        # edge attention map
        att = self.edge_conv(att_map)  # (B,1,H,W)

        # channel enhance
        f = self.feat_proj(c)  # (B,C,H,W)

        # apply BAM (broadcast)
        out = f * (1 + att)  # (B,C,H,W)

        return out


class BAM_(nn.Module):
    """Boundary Attention Module 输入: - feat: YOLO 主干或 neck 特征 (B, C, H, W) - edge: 边缘特征 (B, 1 or k, H, W).

    输出:
        - 融合后的特征 (B, C, H, W)
    """

    def __init__(self, feat_channels, edge_channels=1, reduction=8):
        super().__init__()

        # Step1: 把 edge 通道压成 1 通道（保持空间信息）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_channels, 1, 3, padding=1),
            nn.Sigmoid(),  # 归一化
        )

        # Step2: 用 1×1 conv 调整 feat 的 channel，增强表达能力
        self.feat_proj = nn.Conv2d(feat_channels, feat_channels, 1)

    def forward(self, x):
        """feat: (B, C, H, W) edge: (B, k, H, W).
        """
        c = x[0]
        att_map = x[1][1]
        sizes = [att_map["p3"].size()[2:], att_map["p4"].size()[2:], att_map["p5"].size()[2:]]
        for i in range(len(sizes)):
            if c.size()[2:] == sizes[i]:
                att_map = att_map["p" + str(i + 3)]
                break
        # edge attention map
        att = self.edge_conv(att_map)  # (B,1,H,W)

        # channel enhance
        f = self.feat_proj(c)  # (B,C,H,W)

        # apply BAM (broadcast)
        out = f * (1 + att)  # (B,C,H,W)

        return out


class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        x.size()[2:]
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            if m.i == 10:
                predict_edge = x[0]
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x, predict_edge

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=4, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0][0]

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for
        channels, nc (int) for num classes, anchors (list).
        """
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is BAM:
            args = [ch[x] for x in f]
            c2 = args[0]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is edge:
            c2 = args[0]
            args = [[ch[x] for x in f], *args[1:]]
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{f!s:>18}{n_:>3}{np:10.0f}  {t:<40}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
