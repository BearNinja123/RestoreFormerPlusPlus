import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
torch.set_printoptions(sci_mode=False)

from torch.utils.cpp_extension import load
gn_op = load(name="gn_op", sources=["custom_gn.cpp", "custom_gn_kernel.cu"], extra_cuda_cflags=['--extended-lambda', '-lineinfo'])

class GN_NHWCRefRef(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        N, C, H, W = x.shape
        # transpose columns such that norm is G-contiguous (D, G) rather than D-contiguous (G, D)
        x_reshaped = x.contiguous().view(N, C//self.num_groups, self.num_groups, H, W).transpose(1, 2).reshape(N, C, H, W).contiguous()
        return super().forward(x_reshaped).view(N, self.num_groups, C//self.num_groups, H, W).transpose(1, 2).reshape(N, C, H, W).contiguous() # untranspose x

class GN_NHWCRef(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        N, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).view(N, H * W * C//self.num_groups, self.num_groups) # NCHW -> NHWC -> N(HWD)G
        x_norm = (x_reshaped - x_reshaped.mean(1, keepdims=True)) * (x_reshaped.var(1, correction=0, keepdims=True) + self.eps).rsqrt()
        x_reshaped2 = x_norm.view(N, H, W, C).permute(0, 3, 1, 2) # N(HWD)G -> NHWC -> NCHW
        if self.affine:
            return x_reshaped2 * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        else:
            return x_reshaped2


class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
        X_out, means, rstds = gn_op.forward(X, weight, bias, G, eps)
        ctx.save_for_backward(X, weight, means, rstds, torch.Tensor([G]))
        return X_out

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous(memory_format=torch.channels_last)
        X, weight, means, rstds, G = ctx.saved_tensors 
        dx, dgamma, dbeta = gn_op.backward(dy, X, weight, means, rstds, int(G))
        return dx, dgamma, dbeta, None, None

class GN_NHWC(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        if self.affine:
            return GN_NHWC_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NHWC_Func.apply(x, w, b, self.num_groups, self.eps)

if __name__ == '__main__':
    C = 256
    ##x = torch.arange(C).reshape((1, C, 1, 1)).float().cuda().requires_grad_(True)
    #x = torch.randn((1, C, 2, 2)).cuda().requires_grad_(True).contiguous(memory_format=torch.channels_last)
    #gn1 = GN_NHWCRef(4, C).cuda()
    #gn2 = GN_NHWC(4, C).cuda()
    #with torch.no_grad():
    #    w = torch.randn((C,))
    #    b = torch.randn((C,))
    #    gn1.weight.copy_(w)
    #    gn1.bias.copy_(b)
    #    gn2.weight.copy_(w)
    #    gn2.bias.copy_(b)
    #g1 = gn1(x)
    #print('FORWARD')
    #print('g1')
    #print(g1.reshape((g1.numel(),)))
    #g2 = gn2(x)
    #print('g2')
    #print(g2.reshape((g2.numel(),)))
    #print('BACKWARD')
    #print('g1 sum wrt x')
    #print(torch.autograd.grad(g1.sum(), x, retain_graph=True)[0].reshape((x.numel(),)))
    #print('g2 sum wrt x')
    #print(torch.autograd.grad(g2.sum(), x, retain_graph=True)[0].reshape((x.numel(),)))

    #print('g1 sum wrt w')
    #print(torch.autograd.grad(g1.sum(), gn1.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),)))
    #print('g2 sum wrt w')
    #print(torch.autograd.grad(g2.sum(), gn2.weight, retain_graph=True)[0].reshape((gn2.weight.numel(),)))
    #print('g1 sum wrt b')
    #print(torch.autograd.grad(g1.sum(), gn1.bias, retain_graph=True)[0].reshape((gn1.bias.numel(),)))
    #print('g2 sum wrt b')
    #print(torch.autograd.grad(g2.sum(), gn2.bias, retain_graph=True)[0].reshape((gn2.bias.numel(),)))

    x_nchw = torch.randn((32, C, 128, 128)).cuda().requires_grad_(True)
    x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)
    #gn_nchw = nn.GroupNorm(*gn_args).cuda()
    #gn1 = GN_NHWCRef(*gn_args).cuda()
    #gn2 = GN_NHWC(*gn_args).cuda()
    gn_args = (8, C)
    for gn_class, gn_input, desc in (
            (GN_NHWC, x_nhwc, 'GN NHWC (custom op)'),
            #(nn.GroupNorm, x_nchw, 'nn GN NCHW'),
            #(nn.GroupNorm, x_nhwc, 'nn GN NHWC'),
            #(GN_NHWCRef, x_nhwc, 'GN NHWC (reference)'),
            ):
        print(desc)
        gn_layer = gn_class(*gn_args).cuda()
        for i in tqdm(range(1), smoothing=1):
            g = gn_layer(gn_input)
            #torch.autograd.grad(g.sum(), gn_input)
            torch.cuda.synchronize()
