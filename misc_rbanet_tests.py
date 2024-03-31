from RestoreFormer.modules.vqvae.vqvae_arch import RBAEncoder, RBANet, VQVAEGAN, MultiHeadEncoder, VQVAEGANMultiHeadTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

OPTION = 0
if OPTION == 0:
    with torch.no_grad():
        x = torch.randn((1,3,512, 512)).cuda().bfloat16()
        #m = RBANet(ch_mult=(1,2,4,4)).cuda().bfloat16()
        #m2 = VQVAEGAN(ch_mult=(1,2,4,4)).cuda().bfloat16()
        m = RBANet().cuda().bfloat16()
        m2 = VQVAEGAN().cuda().bfloat16()

        for _ in range(10):
            y = m(x, x)[0]
            torch.cuda.synchronize()
        for _ in tqdm(range(30)):
            y = m(x, x)[0]
            torch.cuda.synchronize()
        for _ in range(10):
            y = m2(x)[0]
            torch.cuda.synchronize()
        for _ in tqdm(range(30)):
            y = m2(x)[0]
            torch.cuda.synchronize()

        fig, ax = plt.subplots(1, 2)
        y = m(x, x)[0]
        ax[0].imshow(y[0].permute(1, 2, 0).cpu().float())
        y = m2(x)[0]
        ax[1].imshow(y[0].permute(1, 2, 0).cpu().float())
        plt.show()
elif OPTION == 1:
    m = VQVAEGANMultiHeadTransformer()
    m2 = RBANet()
    sd = m.state_dict()
    print(len(m.state_dict().keys()))
    print('vqvaegan nparams:', sum(p.numel() for p in m.parameters()))
    print(len(m2.state_dict().keys()))
    print('rba nparams:', sum(p.numel() for p in m2.parameters()))
    m2.load_state_dict(sd, strict=False)