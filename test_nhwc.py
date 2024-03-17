from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGAN, NORM, GN_Normalize
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn as nn
import torch
import time

print(NORM)
torch.random.manual_seed(0)

DTYPE = torch.bfloat16
m_ = VQVAEGAN(attn_resolutions=[16], enable_mid=True).to('cuda', dtype=DTYPE)
#print(sum(p.numel() for p in m_.parameters()))
x = torch.rand((2, 3, 512, 512), device='cuda', dtype=DTYPE)

BWD = False
COMPILE = False
ctx = nullcontext() if BWD else torch.no_grad()
comp = lambda m: torch.compile(m,
        mode='reduce-overhead'
        #backend='cudagraphs',
        #fullgraph=True,
        )

NWARMUP = 5
NTRIAL = 20
with ctx:
    if 'NN' in NORM:
        if COMPILE:
            m = comp(m_)
        else:
            m = m_
        for i in tqdm(range(NWARMUP), smoothing=1):
            y = m(x)[0]
            if BWD:
                y.sum().backward()
        torch.cuda.synchronize()
        tic = time.time()
        for i in tqdm(range(NTRIAL), smoothing=1):
            y = m(x)[0]
            if BWD:
                y.sum().backward()
        torch.cuda.synchronize()
        print('nchw it/s', NTRIAL / (time.time() - tic))

    else:
        m = m_.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
        if COMPILE:
            m = comp(m)
        else:
            m = m
        for i in tqdm(range(NWARMUP), smoothing=1):
            y = m(x)[0]
            if BWD:
                y.sum().backward()
        torch.cuda.synchronize()
        tic = time.time()
        for i in tqdm(range(NTRIAL), smoothing=1):
            y = m(x)[0]
            if BWD:
                y.sum().backward()
        torch.cuda.synchronize()
        print('nhwc it/s', NTRIAL / (time.time() - tic))
