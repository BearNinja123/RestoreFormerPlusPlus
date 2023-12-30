import torch.nn.functional as F
import torch.nn as nn
import torch
torch.set_printoptions(sci_mode=False)

class GN_NHWCNaiveNaive(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        N, C, H, W = x.shape
        # transpose columns such that norm is G-contiguous (D, G) rather than D-contiguous (G, D)
        x_reshaped = x.contiguous().view(N, C//self.num_groups, self.num_groups, H, W).transpose(1, 2).reshape(N, C, H, W).contiguous()
        return super().forward(x_reshaped).view(N, self.num_groups, C//self.num_groups, H, W).transpose(1, 2).reshape(N, C, H, W).contiguous() # untranspose x

class GN_NHWCNaive(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        N, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).view(N, H * W * C//self.num_groups, self.num_groups) # NCHW -> NHWC -> N(HWD)G
        x_norm = (x_reshaped - x_reshaped.mean(1, keepdims=True)) * (x_reshaped.var(1, correction=0, keepdims=True) + self.eps).rsqrt()
        x_reshaped2 = x_norm.view(N, H, W, C).permute(0, 3, 1, 2) # N(HWD)G -> NHWC -> NCHW
        return x_reshaped2 * self.weight[None, :, None, None] + self.bias[None, :, None, None]

if __name__ == '__main__':
    x = 100*torch.randn((2, 64, 1, 1)).to(memory_format=torch.channels_last).cuda() + 13
    gn1 = GN_NHWCNaive(8, 64).cuda()
    gn2 = GN_NHWC(8, 64).cuda()
    g1, g2 = gn1(x), gn2(x)
    print((g1 - g2).abs().mean())
