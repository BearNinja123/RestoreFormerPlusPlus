from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGAN, NORM
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn as nn
import torch
import time

DTYPE = torch.bfloat16
m_ = VQVAEGAN(attn_resolutions=[16], enable_mid=True).to('cuda', dtype=DTYPE)
#print(sum(p.numel() for p in m_.parameters()))
print(NORM)
x = torch.rand((1, 3, 256, 256), device='cuda', dtype=DTYPE)

COMPILE = False
BWD = False
ctx = nullcontext() if BWD else torch.no_grad()
comp = lambda m: torch.compile(m,
        #mode='reduce-overhead',
        #fullgraph=True,
        )

NWARMUP = 15
NTRIAL = 50
with ctx:
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
    print('nchw', time.time() - tic)

    m = m.to(memory_format=torch.channels_last)
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
    print('nhwc', time.time() - tic)
