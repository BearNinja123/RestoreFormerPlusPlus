import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
torch.set_printoptions(sci_mode=False)

print('Started loading module')
from torch.utils.cpp_extension import load
gn_op = load(name="gn_op", sources=["custom_gn.cpp", "custom_gn_kernel.cu", "nchw_kernel.cu"], extra_cuda_cflags=['--extended-lambda', '-use_fast_math', '-lineinfo'])
print('Finished loading module')

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

class GN_NCHW_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
        X_out, means, rstds = gn_op.nchwforward(X, weight, bias, G, eps)
        ctx.save_for_backward(X, weight, means, rstds, torch.Tensor([G]))
        return X_out

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        X, weight, means, rstds, G = ctx.saved_tensors 
        dx, dgamma, dbeta = gn_op.nchwbackward(dy, X, weight, means, rstds, int(G))
        return dx, dgamma, dbeta, None, None

class GN_NCHW(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        if self.affine:
            return GN_NCHW_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NCHW_Func.apply(x, w, b, self.num_groups, self.eps)

if __name__ == '__main__':
    '''
    fwd (nhwc, nchw):
    fp32: 164, 157
    fp16: 225, 206
    bf16: 233, 212

    bwd (nhwc, nchw):
    fp32: 
    fp16: 
    bf16: 
    '''
    C = 256
    DTYPE = torch.float
    MODE = 'bench' # can be 'check', 'bench', default does both

    if MODE != 'bench':
        #x = torch.arange(C).reshape((1, C, 1, 1)).float().cuda().requires_grad_(True)
        #torch.random.manual_seed(0)

        x = torch.randn((2, C, 4, 4), dtype=DTYPE).cuda().requires_grad_(True).contiguous(memory_format=torch.channels_last) #* 100
        gn1 = GN_NHWCRef(4, C).cuda().to(DTYPE)
        gn2 = GN_NHWC(4, C).cuda().to(DTYPE)

        #x = torch.randn((2, C, 4, 4), dtype=DTYPE).cuda().requires_grad_(True) * 100
        #gn1 = nn.GroupNorm(4, C).cuda().to(DTYPE)
        #gn2 = GN_NCHW(4, C).cuda().to(DTYPE)

        with torch.no_grad():
            w = torch.randn((C,))
            b = torch.randn((C,))
            gn1.weight.copy_(w)
            gn1.bias.copy_(b)
            gn2.weight.copy_(w)
            gn2.bias.copy_(b)
        g1 = gn1(x)
        g2 = gn2(x)
        print('FORWARD')
        print('g1', g1.shape)
        print((g1-g2).reshape((g1.numel(),)))
        print('BACKWARD')
        print('g1 sum wrt x')
        g1_grad_wrt_x = torch.autograd.grad(g1.sum(), x, retain_graph=True)[0] #.reshape((x.numel(),))
        g2_grad_wrt_x = torch.autograd.grad(g2.sum(), x, retain_graph=True)[0] #.reshape((x.numel(),))
        print(g1_grad_wrt_x-g2_grad_wrt_x)

        print('g1 sum wrt w')
        g1_grad_wrt_w = torch.autograd.grad(g1.sum(), gn1.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        g2_grad_wrt_w = torch.autograd.grad(g2.sum(), gn2.weight, retain_graph=True)[0].reshape((gn1.weight.numel(),))
        print(g1_grad_wrt_w - g2_grad_wrt_w)

        print('g1 sum wrt b')
        g1_grad_wrt_b = torch.autograd.grad(g1.sum(), gn1.bias, retain_graph=True)[0].reshape((gn1.bias.numel(),))
        g2_grad_wrt_b = torch.autograd.grad(g2.sum(), gn2.bias, retain_graph=True)[0].reshape((gn2.bias.numel(),))
        print(g1_grad_wrt_b - g2_grad_wrt_b)

    if MODE != 'check':
        x_nchw = torch.randn((32, C, 128, 128), dtype=DTYPE, device='cuda').requires_grad_(True)
        x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)
        gn_args = (32, C)
        BENCH = 'both' # can be 'fwd', 'bwd', anything else is fwd + bwd
        for gn_class, gn_input, desc in (
                (GN_NHWC, x_nhwc, 'GN NHWC (custom op)'),
                #(GN_NCHW, x_nchw, 'GN NCHW'),
                #(nn.GroupNorm, x_nchw, 'nn GN NCHW'),
                (nn.GroupNorm, x_nhwc, 'nn GN NHWC'),
                #(GN_NHWCRef, x_nhwc, 'GN NHWC (reference)'),
                ):
            print(desc, BENCH)
            gn_layer = gn_class(*gn_args).cuda().to(DTYPE)
            g = gn_layer(gn_input)
            for i in tqdm(range(100)):
                if BENCH != 'bwd':
                    g = gn_layer(gn_input)
                if BENCH != 'fwd':
                    torch.autograd.grad(g.sum(), gn_input, retain_graph=True)
                torch.cuda.synchronize()
