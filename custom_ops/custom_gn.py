import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time, os
torch.set_printoptions(sci_mode=False)
module_dir = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load
gn_op = load(
        name="gn_op",
        sources=[
            os.path.join(module_dir, "custom_gn.cpp"),
            os.path.join(module_dir, "custom_gn_bwd_kernel.cu"),
            os.path.join(module_dir, "custom_gn_kernel.cu"),
            os.path.join(module_dir, "custom_gn_kernel2.cu"),
            os.path.join(module_dir, "custom_gn_kernel3.cu"),
            os.path.join(module_dir, "custom_gn_kernel4.cu"),
            os.path.join(module_dir, "nchw_kernel.cu")
            ],
        extra_cuda_cflags=[
            '--extended-lambda',
            '-use_fast_math',
            '-lineinfo'
            ],
        extra_cflags=['-O3'], # needed or else GN NCHW from source is slower than nn.GroupNorm
        #verbose=True
        )

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
        #X_out, means, rstds = gn_op.forward4(X, weight, bias, G, eps)
        #if X.shape[0] <= 4 and weight.shape[0] / G >= 4:
        #if X.shape == (4, 64, 256, 256):
        #print(X.shape, G)
        if X.shape[0] <= 8 and X.shape[1] / G < 8 and weight.dtype in (torch.bfloat16, torch.half):
            X_out, means, rstds = gn_op.forward4(X, weight, bias, G, eps)
        elif X.shape[0] <= 8:
            X_out, means, rstds = gn_op.forward2(X, weight, bias, G, eps)
        else:
            X_out, means, rstds = gn_op.forward3(X, weight, bias, G, eps)
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
        #print(x.shape, self.num_channels)
        if x[0].numel() % 512 != 0:
            raise ValueError(f'X[0] has shape {x[0].shape} which is not a multiple of 512. This input is not supported.')
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
    C = 512
    DTYPE = torch.bfloat16
    print('DTYPE:', DTYPE)
    MODE = 'bench' # can be 'check', 'bench', default does both

    if MODE != 'bench':
        #x = torch.arange(C).reshape((1, C, 1, 1)).float().cuda().requires_grad_(True)
        #torch.random.manual_seed(0)

        x = torch.randn((2, C, 4, 4), dtype=DTYPE).cuda().requires_grad_(True).contiguous(memory_format=torch.channels_last) #* 100
        #gn1 = GN_NHWCRef(4, C).cuda().to(DTYPE)
        #gn1 = nn.GroupNorm(4, C).cuda().to(DTYPE)
        #gn2 = GN_NHWC(4, C).cuda().to(DTYPE)
        #gn1 = nn.GroupNorm(C//4, C).cuda().to(DTYPE)
        #gn2 = GN_NHWC(C//4, C).cuda().to(DTYPE)

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
        print('g2', g1.shape)
        #print(g1.reshape(g1.numel()))
        #print(g2.reshape(g1.numel()))
        print((g1-g2).reshape(g1.numel()))
        print(g2-g2)
        raise
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
        #for B, C, R, NTRIALS in [(4, 512, 8, 10000), (8, 256, 128, 1000)]:
        for B, C, R, NTRIALS in [
                #(32, 64, 256, 100),
                #(32, 256, 64, 1),
                #(4, 256, 32, 1),
                #(32, 256, 64, 1000),
                #(1, 256, 32, 40000),
                #(4, 256, 32, 10000),
                #(32, 256, 32, 1000),

                #(1, 64, 256, 1000),
                #(1, 128, 128, 2000),
                #(1, 256, 64, 5000),
                #(1, 256, 32, 10000),
                #(1, 256, 16, 20000),
                #(1, 512, 8, 20000),
                #(4, 64, 256, 1000),
                #(4, 128, 128, 2000),
                #(4, 256, 64, 5000),
                #(4, 256, 32, 10000),
                #(4, 256, 16, 20000),
                #(4, 512, 8, 20000),
                #(8, 64, 256, 500),
                #(8, 128, 128, 1000),
                #(8, 256, 64, 2000),
                #(8, 256, 32, 5000),
                #(8, 256, 16, 10000),
                #(8, 512, 8, 10000),
                #(32, 64, 256, 100),
                #(32, 128, 128, 200),
                #(32, 256, 64, 500),
                #(32, 256, 32, 1000),
                #(32, 256, 16, 4000),
                #(32, 512, 8, 4000),

                #(4, 64, 8, 1),
                #(4, 64, 16, 1),
                #(4, 64, 12, 1),
                #(4, 64, 64, 1),
                #(4, 64, 128, 1),
                #(4, 64, 256, 1),
                #(4, 256, 8, 1),
                #(4, 256, 16, 1),
                #(4, 256, 12, 1),
                #(4, 256, 64, 1),
                #(4, 256, 128, 1),
                #(4, 256, 256, 1),
                (4, 64, 8, 10000),
                (4, 64, 16, 10000),
                (4, 64, 32, 10000),
                (4, 256, 8, 10000),
                (4, 256, 16, 10000),
                (4, 256, 32, 10000),
                (4, 256, 64, 10000),
                (4, 256, 128, 1000),
                (4, 256, 256, 1000),
                #(4, 128, 128, 1),
                #(4, 256, 64, 1),
                #(4, 256, 32, 1),
                #(4, 256, 16, 1),
                #(4, 512, 8, 1),
                ]:
            #x_nchw = torch.randn((32, C, 128, 128), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)
            gn_args = (32, C)
            #gn_args = (C,)
            BENCH = 'fwd' # can be 'fwd', 'bwd', anything else is fwd + bwd
            print(BENCH, x_nchw.shape)
            for gn_class, gn_input, desc, fwd_fn in (
                    (GN_NHWC, x_nhwc, 'GN NHWC4 (custom op)', gn_op.forward4),
                    (GN_NHWC, x_nhwc, 'GN NHWC3 (custom op)', gn_op.forward3),
                    (GN_NHWC, x_nhwc, 'GN NHWC2 (custom op)', gn_op.forward2),
                    #(GN_NHWC, x_nhwc, 'GN NHWC (custom op)', gn_op.forward),
                    #(GN_NCHW, x_nchw, 'nn GN NCHW (from src)', None),
                    #(nn.GroupNorm, x_nchw, 'nn GN NCHW'),
                    #(nn.GroupNorm, x_nhwc, 'nn GN NHWC'),
                    #(GN_NHWCRef, x_nhwc, 'GN NHWC (reference)'),
                    #(nn.BatchNorm2d, x_nhwc, 'BN NHWC'),
                    #(nn.BatchNorm2d, x_nchw, 'BN NCHW'),
                    ):
                print(desc)
                gn_layer = gn_class(*gn_args).cuda().to(DTYPE)
                #g = gn_layer(gn_input)
                torch.cuda.synchronize()
                for i in tqdm(range(NTRIALS)):
                    if BENCH != 'bwd':
                        if isinstance(gn_layer, GN_NHWC):
                            g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                        elif isinstance(gn_layer, GN_NCHW):
                            g = gn_op.nchwforward(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps)
                        else:
                            g = gn_layer(gn_input)
                    if BENCH != 'fwd':
                        if 'NHWC' in desc:
                            g_mem_fmt = g.contiguous(memory_format=torch.channels_last) # in NHWC models, must convert possibly NCHW outputs into NHWC (i.e. from nn GN), note that this is a no-op if g is already in NHWC format (e.g. GN_NHWC output)
                        else:
                            g_mem_fmt = g.contiguous()
                        torch.autograd.grad(g_mem_fmt.sum(), gn_input, retain_graph=True)
                    torch.cuda.synchronize()
            print()
