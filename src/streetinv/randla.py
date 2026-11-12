"""RandLA-Net (Hu et al., CVPR 2020) in plain PyTorch - no custom CUDA ops.

Neighbour / sub-sampling / up-sampling indices are pre-computed on the CPU
(see dataset.build_pyramid), so the network only needs gather + max.
Tensors are laid out as (B, N, C).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cin, cout, bn=True, act=True):
        super().__init__()
        self.lin = nn.Linear(cin, cout, bias=not bn)
        self.bn = nn.BatchNorm1d(cout) if bn else None
        self.act = act

    def forward(self, x):
        x = self.lin(x)
        if self.bn is not None:
            shp = x.shape
            x = self.bn(x.reshape(-1, shp[-1])).reshape(shp)
        return F.leaky_relu(x, 0.2) if self.act else x


def gather(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feats (B, N, C), idx (B, M, K) -> (B, M, K, C)."""
    B, M, K = idx.shape
    C = feats.shape[-1]
    out = torch.gather(feats, 1, idx.reshape(B, M * K, 1).expand(-1, -1, C))
    return out.reshape(B, M, K, C)


class AttentivePooling(nn.Module):
    def __init__(self, d, dout):
        super().__init__()
        self.score = nn.Linear(d, d, bias=False)
        self.mlp = MLP(d, dout)

    def forward(self, x):  # (B, N, K, d)
        w = torch.softmax(self.score(x), dim=2)
        return self.mlp((x * w).sum(2))


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pos1 = MLP(10, d // 2)
        self.pool1 = AttentivePooling(d, d // 2)
        self.pos2 = MLP(d // 2, d // 2)
        self.pool2 = AttentivePooling(d, d)

    def forward(self, xyz, f, nidx):
        nxyz = gather(xyz, nidx)
        cxyz = xyz.unsqueeze(2).expand_as(nxyz)
        rel = cxyz - nxyz
        pe = self.pos1(torch.cat([rel.norm(dim=-1, keepdim=True), rel, cxyz, nxyz], -1))
        f = self.pool1(torch.cat([gather(f, nidx), pe], -1))
        pe = self.pos2(pe)
        return self.pool2(torch.cat([gather(f, nidx), pe], -1))


class DilatedResidualBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.mlp1 = MLP(din, dout // 2)
        self.lfa = LocalFeatureAggregation(dout)
        self.mlp2 = MLP(dout, 2 * dout, act=False)
        self.shortcut = MLP(din, 2 * dout, act=False)

    def forward(self, xyz, f, nidx):
        x = self.lfa(xyz, self.mlp1(f), nidx)
        return F.leaky_relu(self.mlp2(x) + self.shortcut(f), 0.2)


class RandLANet(nn.Module):
    def __init__(self, d_in: int, num_classes: int, d_out=(16, 64, 128, 256)):
        super().__init__()
        self.fc0 = MLP(d_in, 8)
        self.encoder = nn.ModuleList()
        din, skip_dims = 8, []
        for i, d in enumerate(d_out):
            self.encoder.append(DilatedResidualBlock(din, d))
            din = 2 * d
            if i == 0:
                skip_dims.append(din)
            skip_dims.append(din)
        self.mid = MLP(din, din)
        self.decoder = nn.ModuleList()
        f = din
        for j in range(len(d_out)):
            s = skip_dims[-j - 2]
            self.decoder.append(MLP(f + s, s))
            f = s
        self.head = nn.Sequential(MLP(f, 64), MLP(64, 32), nn.Dropout(0.5), nn.Linear(32, num_classes))

    def forward(self, batch: dict) -> torch.Tensor:
        f = self.fc0(batch["feats"])
        skips = []
        for i, layer in enumerate(self.encoder):
            fe = layer(batch["xyz"][i], f, batch["neigh"][i])
            fs = gather(fe, batch["pool"][i]).max(dim=2).values
            if i == 0:
                skips.append(fe)
            skips.append(fs)
            f = fs
        f = self.mid(skips[-1])
        for j, dec in enumerate(self.decoder):
            fi = gather(f, batch["up"][-j - 1]).squeeze(2)
            f = dec(torch.cat([skips[-j - 2], fi], -1))
        return self.head(f)  # (B, N, num_classes)
