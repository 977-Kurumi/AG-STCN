import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        padding = (0, (kernel_size - 1) * dilation)
        super(CausalConv2d, self).__init__(in_channels, out_channels, (1, kernel_size),
                              dilation=dilation, padding=padding, **kwargs)
    def forward(self, x):
        out = super(CausalConv2d, self).forward(x)
        return out[:, :, :, :-self.padding[1]]


class CausalGLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,dilation=1):
        super(CausalGLUBlock, self).__init__()
        self.conv_A = CausalConv2d(in_channels, out_channels, kernel_size)
        self.conv_B = CausalConv2d(in_channels, out_channels, kernel_size)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, (1, 1)
                              ) if in_channels != out_channels else None

    def forward(self, X):
        A = self.conv_A(X)
        B = self.conv_B(X)
        output = A * torch.sigmoid(B)

        return output


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TimeBlock, self).__init__()
        self.glu1 = CausalGLUBlock(in_channels, out_channels, kernel_size)
        self.self_attention = SelfAttention(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1) if in_channels != out_channels else None

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        out1 = self.glu1(X)
        out_final = self.self_attention(out1)
        out_r = out_final.permute(0, 2, 3, 1)
        return out_r


