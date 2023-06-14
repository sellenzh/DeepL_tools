"""
author: @Sellen Prye
date: 2023-06-14
description: BEF used to replace the feedforward layer in the transformer.
mentioned: https://github.com/liuzwin98/DSCMT in DSCMT.py
"""
from torch import nn

## this part is written by Original author
# class BEF(nn.Module):
#     def __init__(self, dimentions, reduction=8):
#         super(BEF, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.layers = nn.Sequential(
#             nn.Linear(dimentions, dimentions // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(dimentions // reduction, dimentions, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         b, c, f = x.size()
#         gap = self.avg_pool(x).view(b, c)
#         y = self.layers(gap).view(b, c, 1)
#         return (x * y.expand_as(x)).contiguous().permute(0, 2, 1).contiguous()

## this part is written by @Sellen Prye
class BEF(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(BEF, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.avgpool(x.transpose(1, 2)).transpose(1, 2)
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        x = self.sig(self.dropout(x))
        x = x * residual
        return x