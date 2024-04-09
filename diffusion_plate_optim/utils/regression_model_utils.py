import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mean_from_field_solution(field_solution, field_mean, field_std, n_frequencies=300):
    v_ref = 1e-9
    eps = 1e-12
    shape = field_solution.shape
    n_frequencies = field_mean.shape[1]
    # remove transformations in dataloader
    field_solution = field_solution * field_std + field_mean
    field_solution = torch.exp(field_solution) - eps
    # calculate frequency response
    field_solution = field_solution.view(shape[0], n_frequencies, -1)
    v = torch.mean(field_solution, dim=-1)
    v = 10*torch.log10((v + 1e-12)/v_ref)
    return v.view(shape[0], n_frequencies)


class Film(nn.Module):
    def __init__(self, conditional_dim, projection_dim, **kwargs):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(conditional_dim, projection_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(projection_dim, projection_dim))
        self.bias = nn.Sequential(nn.Linear(conditional_dim, projection_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(projection_dim, projection_dim))

    def forward(self, x, conditional):
        ndim = len(x.shape) - 2
        view_shape = (x.shape[:2]) + (1,) * ndim
        return self.weight(conditional).view(*view_shape) * x + self.bias(conditional).view(*view_shape)



# model adapted from https://github.com/dome272/Diffusion-Models-pytorch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0]*size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, *size)



class LocalNet(torch.nn.Module):
    def __init__(self, c_in=1, conditional=False, n_frequencies=300, freq_batches=5, scaling_factor=32, len_conditional=None, **kwargs):
        super().__init__()
        k = scaling_factor  # 32
        self.inc = DoubleConv(c_in, k)
        self.down0 = nn.Sequential(nn.Conv2d(k,k,3, stride=2, padding=1), nn.ReLU())

        self.down1 = Down(k, 2*k)
        self.down2 = Down(2*k, 4*k)
        self.sa2 = SelfAttention(4*k)
        self.down3 = Down(4*k, 4*k)
        self.sa3 = SelfAttention(4*k)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(len_conditional, 4*k)
        self.bot1 = DoubleConv(4*k, 4*k)
        self.bot3 = DoubleConv(4*k, 4*k)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.queryfilm = Film(1, 8*k)
        self.up1 = Up(12*k, 4*k)
        self.sa4 = SelfAttention(4*k)
        self.up2 = Up(6*k, 4*k)
        self.up3 = Up(5*k, 2*k)
        self.outc = nn.Conv2d(2*k, freq_batches, kernel_size=1)
        self.freq_batches = freq_batches
        self.freq_steps = int(n_frequencies / freq_batches)
        self.out_dim = n_frequencies
        self.query_frequencies = (torch.arange(0, self.freq_steps) / 59 * 2) - 1 # 60 freq steps for 300 freqs
        self.query_frequencies = self.query_frequencies.cuda()

    def forward(self, x, conditional=None):
        B = x.shape[0]
        x = torch.nn.functional.interpolate(x, size=(96, 128), mode='bilinear', align_corners=True)
        x = self.inc(x)
        x1 = self.down0(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        if self.conditional is True:
            x3 = self.film(x3, conditional)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot3(x4)
        gap = self.global_avg_pool(x4)
        x4 = torch.cat([x4, gap.repeat(1, 1, x4.size(2), x4.size(3))], dim=1)

        x4 = x4.repeat_interleave(len(self.query_frequencies), dim=0)
        queries = self.query_frequencies
        qf = queries.repeat(B).view(-1, 1).expand(B*self.freq_steps, 1)
        x4 = self.queryfilm(x4, qf)
        x = self.up1(x4, x3.repeat_interleave(self.freq_steps, dim=0))
        x = self.sa4(x)
        x = self.up2(x, x2.repeat_interleave(self.freq_steps, dim=0))
        x = self.up3(x, x1.repeat_interleave(self.freq_steps, dim=0))
        output = self.outc(x)
        output = torch.nn.functional.interpolate(output, size=(40, 60), mode='bilinear', align_corners=True)
        return output.reshape(B, self.freq_steps*self.freq_batches, *output.shape[2:])

def get_net():
    return LocalNet(scaling_factor=32)