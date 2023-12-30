from basicsr.archs.stylegan2_arch import UpFirDnUpsample
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True) # apparently this is as fast as flash attn but more flexible

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
        z_q = z_q.permute(0, 3, 1, 2) #.contiguous(memory_format=torch.channels_last)
        return z_q, loss, (None, None, min_encoding_indices, d)

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2) #.contiguous(memory_format=torch.channels_last)

        return z_q

nonlinearity = F.silu

class Normalize(nn.Module): # runs BatchNorm in FP32 because of float16 stability issues when x is large but with small variance (i.e. x = 100) 
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-6, affine=True)
        #self.norm = nn.GroupNorm(num_groups, in_channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            return self.norm(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        #self.upsample = UpFirDnUpsample((1, 3, 3, 1))
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)

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
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, stride=2)

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

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h

class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.k = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.v = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1)

    def split_heads(self, x):
        B, _C, H, W = x.shape
        # B (hD) H W -> B h D (HW) -> B h (HW) D, D = d_head (self.att_size), h = num heads (self.head_size)
        return x.view(B, self.head_size, self.att_size, H * W).permute(0, 3, 1, 2) #.contiguous()
    
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        kv = self.norm1(x)
        q = kv if y is None else self.norm2(y)
        q, k, v = map(self.split_heads, (self.q(q), self.k(kv), self.v(kv)))

        # compute attention
        att = F.scaled_dot_product_attention(q, k, v)
        catted = att.transpose(2, 3).reshape(B, C, H, W) #.contiguous(memory_format=torch.channels_last) # M H N D -> M H D N -> B C H W

        return x + self.proj_out(catted)

class MultiHeadEncoder(nn.Module):
    def __init__(self, ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, double_z=True, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, 3, padding=1)
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

        # end
        self.norm_out = Normalize(block_out)
        self.conv_out = torch.nn.Conv2d(block_out, 2 * z_channels if double_z else z_channels, 3, padding=1)

    def forward(self, x):
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

            if i_level != self.num_resolutions-1:
                hs['block_'+str(i_level)] = h
                h = self.down[i_level].downsample(h)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs['block_'+str(i_level)+'_atten'] = h
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs['mid_atten'] = h

        # end
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        hs['out'] = h

        return hs

class MultiHeadDecoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        block_out = ch * ch_mult[-1]
        self.conv_in = torch.nn.Conv2d(z_channels, block_out, 3, padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(block_out, block_out, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_out, head_size)
            self.mid.block_2 = ResnetBlock(block_out, block_out, dropout=dropout)

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
            if layer_idx != self.num_resolutions - 1:
                up.upsample = Upsample(block_out, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_out)
        self.conv_out = torch.nn.Conv2d(block_out, out_ch, 3, padding=1)

    def forward(self, z, hs=None):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)

        # upsampling
        for rev_i_level, up_level in enumerate(reversed(self.up)):
            i_level = self.num_resolutions - rev_i_level - 1
            for i_block in range(self.num_res_blocks+1):
                h = up_level.block[i_block](h, temb)
                if len(up_level.attn) > 0:
                    if hs is None:
                        h = up_level.attn[i_block](h)
                    elif 'block_'+str(i_level)+'_atten' in hs: # at this point hs is defined
                        h = up_level.attn[i_block](h, hs['block_'+str(i_level)+'_atten'])
                    else:
                        h = up_level.attn[i_block](h, hs['block_'+str(i_level)])
            
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
    def __init__(self,
                 n_embed=1024,
                 embed_dim=256,
                 ch=64,
                 out_ch=3,
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
                 ex_multi_scale_num=1):
        super(VQVAEGANMultiHeadTransformer, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z, 
                               enable_mid=enable_mid, head_size=head_size)
        for i in range(ex_multi_scale_num):
                attn_resolutions = [attn_resolutions[0], attn_resolutions[-1]*2]
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
	    

    def encode(self, x):
        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs=None):
        quant = self.post_quant_conv(quant)
        dec, penult = self.decoder(quant, hs)
        return dec, penult

    def forward(self, input, ref=None):
        quant, diff, info, hs = self.encode(input)
        if ref is not None:
            quant2, _diff, _info, _hs = self.encode(ref)
            quant = quant2
        dec, penult = self.decode(quant, hs)

        return dec, diff, info, hs, penult

class VQVAEGAN(VQVAEGANMultiHeadTransformer): # VQVAEGANMultiHeadTransformer but fix_encoder=False and ex_multi_scale_num=0
    # note: default values of init args are the same as VQVAEGANMultiHeadTransformer
    def __init__(self,
                 n_embed=1024,
                 embed_dim=256,
                 ch=64,
                 out_ch=3,
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
                 **ignore_kwargs):
        super(VQVAEGAN, self).__init__(
            n_embed, embed_dim, ch, out_ch, ch_mult, num_res_blocks,
            attn_resolutions, dropout, in_channels, resolution,
            z_channels, double_z, enable_mid, fix_decoder, fix_codebook,
            False, head_size, 0
            )

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec, penult = self.decoder(quant)
        return dec, penult

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec, penult = self.decode(quant)
        return dec, diff, info, hs, penult