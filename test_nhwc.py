from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGAN
from tqdm import tqdm
import torch.nn as nn
import torch
import time

print(torch._dynamo.list_backends(None))
print(torch._dynamo.list_backends())

m_ = VQVAEGAN(attn_resolutions=[16], enable_mid=True).to('cuda', dtype=torch.bfloat16)
print(sum(p.numel() for p in m_.parameters()))
x = torch.rand((8, 3, 512, 512), device='cuda', dtype=torch.bfloat16)

COMPILE = True
comp = lambda m: torch.compile(m,
        #mode='reduce-overhead',
        backend='cudagraphs',
        #fullgraph=True,
        )

with torch.no_grad():
    if COMPILE:
        m = comp(m_)
    else:
        m = m_
    for i in tqdm(range(5), smoothing=1):
        m(x)
        torch.cuda.synchronize()
    tic = time.time()
    for i in tqdm(range(5), smoothing=1):
        m(x)
        torch.cuda.synchronize()
    print('nchw', time.time() - tic)

    m = m.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    if COMPILE:
        m = comp(m)
    else:
        m = m
    for i in tqdm(range(5), smoothing=1):
        m(x)
        torch.cuda.synchronize()
    tic = time.time()
    for i in tqdm(range(5), smoothing=1):
        m(x)
        torch.cuda.synchronize()
    print('nhwc', time.time() - tic)
