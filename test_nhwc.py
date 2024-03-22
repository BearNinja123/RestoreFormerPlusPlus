from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGAN, NORM, MEM_FMT, GN_Normalize
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn as nn
import torch
import time

print(NORM)
torch.random.manual_seed(0)

DTYPE = torch.bfloat16
m_ = VQVAEGAN().to('cuda', dtype=DTYPE)
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
    m = m_.to(memory_format=MEM_FMT)
    x = x.to(memory_format=MEM_FMT)
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
