from gnNHWC.custom_gn import GN_NHWC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False) # apparently this is as fast as flash attn but more flexible
torch.backends.cudnn.benchmark = True

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1)

        z_flattened = z.reshape(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(1, keepdim=True) + \
            (self.embedding.weight ** 2).sum(1) - 2 * \
            (z_flattened @ self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2) #.contiguous(memory_format=MEM_FMT)
        return z_q, loss, (None, None, min_encoding_indices, d)

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2) #.contiguous(memory_format=MEM_FMT)

        return z_q

NORM = 'GN'
if NORM == 'GN':
    nonlinearity = lambda x: x # nonlinearity fused into GN kernel
    MEM_FMT = torch.channels_last
else:
    #nonlinearity = F.silu
    nonlinearity = lambda x: F.gelu(x, approximate='tanh')
    MEM_FMT = torch.contiguous_format

class BN_Normalize(nn.BatchNorm2d): # runs BatchNorm in FP32 because of float16 stability issues when x is large but with small variance (i.e. x = 100) 
    def __init__(self, in_channels: int, num_groups: int = 32, **ignorekwargs):
        super().__init__(in_channels)
    
    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            return super().forward(x)

class GN_NN_Normalize(nn.GroupNorm): # runs BatchNorm in FP32 because of float16 stability issues when x is large but with small variance (i.e. x = 100) 
    def __init__(self, in_channels: int, channels_per_group: int = 16, **ignorekwargs):
        super().__init__(in_channels // channels_per_group, in_channels)
    
    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            return super().forward(x)

class GN_Normalize(GN_NHWC): # runs BatchNorm in FP32 because of float16 stability issues when x is large but with small variance (i.e. x = 100) 
    def __init__(self, in_channels: int, channels_per_group: int = 16, activation='identity'):
        super().__init__(in_channels // channels_per_group, in_channels, activation)
    
    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            return super().forward(x)

if NORM == 'BN':
    Normalize = BN_Normalize
elif NORM == 'GN NN':
    Normalize = GN_NN_Normalize
else:
    Normalize = GN_Normalize

class ConvLoRA(nn.Module):
    '''
    Copied from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.merged = False
        self.merge_weights = merge_weights
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

def conv_lora_set_weights(conv_layer: nn.Conv2d) -> ConvLoRA:
    lora_layer = ConvLoRA(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size)
    lora_layer.conv.weight.data = conv_layer.weight.data
    lora_layer.conv.bias.data = conv_layer.bias.data
    return lora_layer

def add_lora_adapters(module: nn.Module, layer_lora_fns={nn.Conv2d: conv_lora_set_weights}):
    '''
    Converts a pretrained model inplace to a model with LoRA adapters.
    module:
        An nn.Module (no LoRA layers) with loaded pretrained weights (i.e. a pretrained model)
    layer_lora_fns:
        a dictionary whose keys are layer classes and whose values represent a function that
        takes in a layer and returns a LoRA layer with the same weights as the input layer
    '''
    for name, child in module.named_children():
        if type(child) in layer_lora_fns:
            lora_module = layer_lora_fns[type(child)](child)
            setattr(module, name, lora_module)
        add_lora_adapters(child, layer_lora_fns) # recursively call the function on the module's children

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        #self.upsample = UpFirDnUpsample((1, 3, 3, 1))
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        #x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2.0) #, mode='bilinear')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, activation='gelu_tanh')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = Normalize(out_channels, activation='gelu_tanh')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h

class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1, y_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'
        y_ch = in_channels if y_channels is None else y_channels

        self.norm1 = Normalize(in_channels)
        if y_channels is not None:
            self.norm2 = Normalize(y_ch)

        self.q = nn.Conv2d(y_ch, in_channels, 1) # kinda strange how the y (cross attention input) goes into the query but this is how RF++ does it
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def split_heads(self, x):
        B, _C, H, W = x.shape
        # B (hD) H W -> B h D (HW) -> B h (HW) D, D = d_head (self.att_size), h = num heads (self.head_size)
        return x.view(B, self.head_size, self.att_size, H * W).transpose(2, 3) #.contiguous()
    
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        kv = self.norm1(x)
        q = kv if y is None else self.norm2(y)
        q, k, v = map(self.split_heads, (self.q(q), self.k(kv), self.v(kv)))

        # compute attention
        att = F.scaled_dot_product_attention(q, k, v)
        catted = att.transpose(2, 3).reshape(B, C, H, W) #.contiguous(memory_format=MEM_FMT) # M H N D -> M H D N -> B C H W

        return x + self.proj_out(catted)

class MultiHeadEncoder(nn.Module): # ref_ch = channel count of the reference embedding
    def __init__(
            self, ch, ref_ch=None, ch_mult=(1,2,4,8), num_res_blocks=2,
            attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
            resolution=512, z_channels=256, double_z=True, enable_mid=True,
            head_size=1, **ignore_kwargs
    ):
        super().__init__()
        self.ch = ch
        self.ref_ch = ref_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, padding=1)
        block_out = self.ch

        curr_res = resolution
        self.down = nn.ModuleList()
        for layer_idx, mult in enumerate(ch_mult):
            block_in, block_out = block_out, ch * mult
            block = [ResnetBlock(block_in, block_out, dropout=dropout)]
            block += [ResnetBlock(block_out, block_out, dropout=dropout) for _ in range(self.num_res_blocks-1)]
            attn = [MultiHeadAttnBlock(block_out, head_size) for _ in range(self.num_res_blocks)] if curr_res in attn_resolutions else []
            down = nn.Module()
            down.block = nn.ModuleList(block)
            down.attn = nn.ModuleList(attn)
            if isinstance(ref_ch, int):
                down.cross_attn = MultiHeadAttnBlock(block_out, head_size, y_channels=ref_ch)
            if layer_idx != self.num_resolutions - 1:
                down.downsample = Downsample(block_out, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(block_out, block_out, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_out, head_size)
            self.mid.block_2 = ResnetBlock(block_out, block_out, dropout=dropout)
            if isinstance(ref_ch, int):
                self.mid.cross_attn = MultiHeadAttnBlock(block_out, head_size, y_channels=ref_ch)

        # end
        self.norm_out = Normalize(block_out, activation='gelu_tanh')
        self.conv_out = nn.Conv2d(block_out, 2 * z_channels if double_z else z_channels, 3, padding=1)

    def forward(self, x, x_ref=None): # x_ref.shape = (N, ref_ch, H, W)
        hs = {}
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        hs['in'] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                if x_ref is not None:
                    h = self.down[i_level].cross_attn[i_block](x_ref, h) # x_ref is K/V, h is Q

            if i_level != self.num_resolutions-1:
                hs['block_'+str(i_level)] = h
                h = self.down[i_level].downsample(h)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs['block_'+str(i_level)+'_atten'] = h
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            if x_ref is not None:
                h = self.mid.cross_attn(x_ref, h)
            hs['mid_atten'] = h

        # end
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        hs['out'] = h

        return hs

class MultiHeadDecoder(nn.Module):
    def __init__(
            self, ch, out_ch, ref_ch=None, ch_mult=(1,2,4,8), num_res_blocks=2,
            attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
            resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
            head_size=1, ex_multi_scale_num=0, **ignorekwargs
    ):
        super().__init__()
        self.ch = ch
        self.ref_ch = ref_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid
        self.ex_multi_scale_num = ex_multi_scale_num

        # compute in_ch_mult, block_in and curr_res at lowest res
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        block_out = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_out, 3, padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(block_out, block_out, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_out, head_size)
            self.mid.block_2 = ResnetBlock(block_out, block_out, dropout=dropout)
            if isinstance(ref_ch, int):
                self.mid.cross_attn = MultiHeadAttnBlock(block_out, head_size, y_channels=ref_ch)

        # upsampling
        self.up = nn.ModuleList()
        for layer_idx, mult in enumerate(reversed(ch_mult)):
            block_in, block_out = block_out, ch * mult
            block = [ResnetBlock(block_in, block_out, dropout=dropout)]
            block += [ResnetBlock(block_out, block_out, dropout=dropout) for _ in range(self.num_res_blocks)]
            attn = [MultiHeadAttnBlock(block_out, head_size) for _ in range(self.num_res_blocks + 1)] if curr_res in attn_resolutions else []
            up = nn.Module()
            up.block = nn.ModuleList(block)
            up.attn = nn.ModuleList(attn)
            if isinstance(ref_ch, int):
                up.cross_attn = MultiHeadAttnBlock(block_out, head_size, y_channels=ref_ch)
            if layer_idx != self.num_resolutions - 1:
                up.upsample = Upsample(block_out, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_out, activation='gelu_tanh')
        self.conv_out = nn.Conv2d(block_out, out_ch, 3, padding=1)

    def forward(self, z, hs=None, x_ref=None):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)
            if x_ref is not None:
                h = self.mid.cross_attn(x_ref, h)

        # upsampling
        for rev_i_level, up_level in enumerate(reversed(self.up)):
            i_level = self.num_resolutions - rev_i_level - 1
            for i_block in range(self.num_res_blocks+1):
                h = up_level.block[i_block](h, temb)
                if len(up_level.attn) > 0:
                    # outer layers of the decoder do not have multi-scale fusion
                    if (hs is None) or (i_level >= self.ex_multi_scale_num):
                        h = up_level.attn[i_block](h)

                    # inner layers have multi-scale fusion
                    elif 'block_'+str(i_level)+'_atten' in hs: # at this point hs is defined
                        h = up_level.attn[i_block](h, hs['block_'+str(i_level)+'_atten'])
                    else:
                        h = up_level.attn[i_block](h, hs['block_'+str(i_level)])

                    if x_ref is not None:
                        h = up_level.cross_attn[i_block](x_ref, h)
            
            if i_level != 0:
                h = up_level.upsample(h)

        # end
        if self.give_pre_end:
            return h

        penult = nonlinearity(self.norm_out(h))
        h = self.conv_out(penult)
        return h, penult

MultiHeadDecoderTransformer = MultiHeadDecoder # for backwards compatibility

class VQVAEGANMultiHeadTransformer(nn.Module):
    def __init__(
            self,
            n_embed=1024,
            embed_dim=256,
            ch=64,
            out_ch=3,
            ref_ch=None,
            ch_mult=(1, 2, 2, 4, 4, 8),
            num_res_blocks=2,
            attn_resolutions=(16, ),
            dropout=0.0,
            in_channels=3,
            resolution=512,
            z_channels=256,
            double_z=False,
            enable_mid=True,
            fix_decoder=False,
            fix_codebook=True,
            fix_encoder=False,
            head_size=4,
            ex_multi_scale_num=1
    ):
        super(VQVAEGANMultiHeadTransformer, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ref_ch=ref_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)

        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

        #for i in range(ex_multi_scale_num):
        #        attn_resolutions = [attn_resolutions[0], attn_resolutions[-1]*2]
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size, ex_multi_scale_num=ex_multi_scale_num)

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
        if fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_decoder:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
	    

    def encode(self, x, x_ref=None):
        hs = self.encoder(x, x_ref=x_ref)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs=None, x_ref=None):
        quant = self.post_quant_conv(quant)
        dec, penult = self.decoder(quant, hs, x_ref=x_ref)
        return dec, penult

    def forward(self, input, x_ref=None):
        quant, diff, info, hs = self.encode(input, x_ref=x_ref)
        dec, penult = self.decode(quant, hs, x_ref=x_ref)

        return dec, diff, info, hs, penult

class VQVAEGAN(VQVAEGANMultiHeadTransformer): # VQVAEGANMultiHeadTransformer but fix_encoder=False and ex_multi_scale_num=0
    # note: default values of init args are the same as VQVAEGANMultiHeadTransformer
    def __init__(
            self,
            n_embed=1024,
            embed_dim=256,
            ch=64,
            out_ch=3,
            ref_ch=None,
            ch_mult=(1, 2, 2, 4, 4, 8),
            num_res_blocks=2,
            attn_resolutions=(16, ),
            dropout=0.0,
            in_channels=3,
            resolution=512,
            z_channels=256,
            double_z=False,
            enable_mid=True,
            fix_decoder=False,
            fix_codebook=True,
            head_size=4,
            **ignore_kwargs
    ):
        super(VQVAEGAN, self).__init__(
            n_embed, embed_dim, ch, out_ch, ref_ch, ch_mult, num_res_blocks,
            attn_resolutions, dropout, in_channels, resolution,
            z_channels, double_z, enable_mid, fix_decoder, fix_codebook,
            False, head_size, 0
            )

    def decode(self, quant, x_ref=None):
        quant = self.post_quant_conv(quant)
        dec, penult = self.decoder(quant, x_ref=x_ref)
        return dec, penult

    def forward(self, input, x_ref=None):
        quant, diff, info, hs = self.encode(input, x_ref=x_ref)
        dec, penult = self.decode(quant, x_ref=x_ref)
        return dec, diff, info, hs, penult
