from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, num_channels: int, num_groups: int = 8) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        groups = min(num_groups, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def _make_act(act: str) -> nn.Module:
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {act}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm: str = "gn",
        act: str = "silu",
        use_act: bool = True,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = _make_norm(norm, out_ch)
        self.act = _make_act(act) if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "gn",
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            dilation=1,
            norm=norm,
            act=act,
            use_act=True,
        )
        self.conv2 = ConvNormAct(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            norm=norm,
            act=act,
            use_act=False,
        )

        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=stride,
                norm=norm,
                act=act,
                use_act=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = _make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.out_act(x)


class SPPF(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool_kernel: int = 5,
        norm: str = "gn",
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = in_ch // 2
        self.reduce = ConvNormAct(in_ch, hidden, kernel_size=1, norm=norm, act=act)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        self.expand = ConvNormAct(hidden * 4, out_ch, kernel_size=1, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.expand(torch.cat([x, y1, y2, y3], dim=1))


class TinyDetBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        channels: Tuple[int, int, int, int, int] = (32, 64, 96, 160, 224),
        norm: str = "gn",
        act: str = "silu",
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        self.stem = nn.Sequential(
            ConvNormAct(in_ch, c1, kernel_size=3, stride=2, norm=norm, act=act),
            ConvNormAct(c1, c1, kernel_size=3, stride=2, norm=norm, act=act),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c2, stride=1, norm=norm, act=act),
            ResidualBlock(c2, c2, stride=1, norm=norm, act=act),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, norm=norm, act=act),
            ResidualBlock(c3, c3, stride=1, norm=norm, act=act),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c3, c4, stride=2, norm=norm, act=act),
            ResidualBlock(c4, c4, stride=1, norm=norm, act=act),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(c4, c5, stride=2, norm=norm, act=act),
            ResidualBlock(c5, c5, stride=1, dilation=2, norm=norm, act=act),
        )

        self.context = SPPF(c5, c5, norm=norm, act=act)
        self.out_channels = {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        c5 = self.context(c5)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class TinyDetFPN(nn.Module):
    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: int = 96,
        norm: str = "gn",
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.lat_c5 = ConvNormAct(in_channels["c5"], out_channels, kernel_size=1, norm=norm, act=act)
        self.lat_c4 = ConvNormAct(in_channels["c4"], out_channels, kernel_size=1, norm=norm, act=act)
        self.lat_c3 = ConvNormAct(in_channels["c3"], out_channels, kernel_size=1, norm=norm, act=act)
        self.lat_c2 = ConvNormAct(in_channels["c2"], out_channels, kernel_size=1, norm=norm, act=act)

        self.smooth_p4 = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, act=act)
        self.smooth_p3 = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, act=act)
        self.smooth_p2 = ConvNormAct(out_channels, out_channels, kernel_size=3, norm=norm, act=act)

        self.out_channels = {"p2": out_channels, "p3": out_channels, "p4": out_channels, "p5": out_channels}

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth_p4(p4)

        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth_p3(p3)

        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth_p2(p2)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class TinyDetEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        backbone_channels: Tuple[int, int, int, int, int] = (32, 64, 96, 160, 224),
        fpn_channels: int = 96,
        norm: str = "gn",
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.backbone = TinyDetBackbone(
            in_ch=in_ch,
            channels=backbone_channels,
            norm=norm,
            act=act,
        )
        self.neck = TinyDetFPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
            norm=norm,
            act=act,
        )
        self.out_channels = self.neck.out_channels

    def forward(
        self,
        x: torch.Tensor,
        return_backbone_feats: bool = False,
    ):
        backbone_feats = self.backbone(x)
        pyramid_feats = self.neck(backbone_feats)

        if return_backbone_feats:
            return pyramid_feats, backbone_feats
        return pyramid_feats
