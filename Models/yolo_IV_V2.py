# -*- coding: utf-8 -*-
"""
Main model file for YOLOv5-Lite-IV.

This file defines the `Model` class, which parses a YAML configuration file
and constructs the dual-stream network. It uses modules defined in `common_IV_V2.py`.

The model takes two inputs (infrared and visible images) and, during training,
returns both the detection outputs and the auxiliary losses from the FusionModules.
"""

import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

# Correctly set up path to import from the 'models' directory.
# This assumes a structure like:
# /my_project
#   /models
#     yolo_IV_V2.py
#     common_IV_V2.py
#   train.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Use a relative import, which is more robust for packages.
# The `*` import is kept for compatibility with the original YOLOv5 parsing logic.
from .common_IV_V2 import *

try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Please run 'pip install pyyaml'")
    sys.exit(1)

LOGGER = logging.getLogger(__name__)


class Detect(nn.Module):
    """Detection layer for YOLOv5. Outputs predictions at 3 different scales."""
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (xywh, obj_conf, class_conf)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    """
    YOLOv5-Lite-IV model. Takes a YAML config file, and channel numbers for
    visible and infrared inputs.
    """

    def __init__(self, cfg='yolov5l_IV_V2.yaml', ch_visible=3, ch_infrared=1, nc=None, anchors=None):
        super().__init__()
        # Load YAML
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)

        # Define model
        self.ch_ir = ch_infrared
        self.ch_vis = ch_visible
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[self.ch_ir, self.ch_vis])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # dummy image size
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.ch_ir, s, s),
                                                                           torch.zeros(1, self.ch_vis, s, s))[0]])
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        # Init weights, biases
        self.initialize_weights()
        LOGGER.info('YOLOv5-Lite-IV model initialized.')

    def forward(self, x_ir, x_vis):
        """
        Forward pass through the network.
        Returns detection outputs and a list of fusion losses during training.
        """
        y = []
        loss_shared_list = []
        loss_unique_list = []

        # The first input (x_ir) is routed to the first backbone branch.
        # The second input (x_vis) is routed to the second backbone branch (layers with from=-2).
        # See the `parse_model` function for how inputs are handled.
        for m in self.model:
            if m.f != -1 and not isinstance(m.f, list):  # Not from previous layer or list of layers
                # Handle specific layer indices
                if m.f == -2:  # Explicitly from visible input
                    x = x_vis
                else:
                    x = y[m.f]
            elif isinstance(m.f, list):  # From a list of layers (e.g., for Concat or FusionModule)
                x = [y[i] for i in m.f]
            else:  # From previous layer (-1)
                if m.i == 0:  # First layer of the model
                    x = x_ir
                else:
                    x = y[-1]

            # Run module
            if isinstance(m, FusionModule):
                fused_feature, loss_shared, loss_unique = m(x)
                x = fused_feature  # Main feature stream
                loss_shared_list.append(loss_shared)
                loss_unique_list.append(loss_unique)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)

        if self.training:
            return x, loss_shared_list, loss_unique_list
        else:
            return x[0] if isinstance(x, tuple) else x

    def initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # Use PyTorch's default initialization
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True


def parse_model(d, ch):
    """
    Parses a model dictionary and builds the layers.
    d: model dictionary from YAML
    ch: a list containing input channels, e.g., [ch_ir, ch_vis]
    """
    # Log model header
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

    # Get params from dict
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    # Init lists
    layers, save = [], []
    ch_list = [0] * (len(d['backbone']) + len(d['head']))  # Store channel count for each layer
    ch_ir, ch_vis = ch[0], ch[1]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # Evaluate module and arguments
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        # Determine input channels
        if isinstance(f, int) and f != -2:
            # Regular 'from' index (-1 means previous layer)
            c1 = ch_ir if i == 0 else ch_list[f]
        elif f == -2:
            # Special index for visible input stream
            c1 = ch_vis
        else:  # A list of 'from' indices (for Concat, FusionModule)
            c1 = sum(ch_list[fi] for fi in f)

        # Define module
        if m in [Conv, C3, SPPF, Bottleneck]:
            c2 = make_divisible(args[0] * gw, 8) if args[0] != no else args[0]
            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [c1]
            c2 = c1
        elif m is Concat:
            c2 = c1
        elif m is FusionModule:
            # Assumes both inputs to FusionModule have the same channel count
            in_ch = ch_list[f[0]]
            out_ch = args[1]
            args = [in_ch, out_ch]
            c2 = out_ch
        elif m is Detect:
            args.append([ch_list[fi] for fi in f])  # pass input channels to Detect layer
            c2 = no
        else:
            c2 = c1

        # Create module sequence
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from', type, and params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))

        save.extend(x for x in ([f] if isinstance(f, int) else f) if x != -1)  # Track layers to save for Concat
        layers.append(m_)
        ch_list[i] = c2

    return nn.Sequential(*layers), sorted(list(set(save)))


def make_divisible(x, divisor):
    """Returns x rounded to the nearest multiple of divisor."""
    return math.ceil(x / divisor) * divisor









































































