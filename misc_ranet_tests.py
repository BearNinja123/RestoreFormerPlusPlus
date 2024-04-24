from RestoreFormer.modules.vqvae.vqvae_arch import RAEncoder, RANet, VQVAEGAN, MultiHeadEncoder, VQVAEGANMultiHeadTransformer, MEM_FMT
from RestoreFormer.models.vqgan_v1 import RAModel
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch

OPTION = 2
if OPTION == 0:
    with torch.no_grad():
        x = torch.randn((1,3,512, 512)).cuda().to(memory_format=MEM_FMT).bfloat16()
        #m = RANet(ch_mult=(1,2,4,4)).cuda().bfloat16()
        #m2 = VQVAEGAN(ch_mult=(1,2,4,4)).cuda().bfloat16()
        m = RANet().cuda().to(memory_format=MEM_FMT).bfloat16()
        m2 = VQVAEGAN().cuda().to(memory_format=MEM_FMT).bfloat16()

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
        yimg = y[0].permute(1, 2, 0).cpu()
        yimg = ((yimg - yimg.min()) / (yimg.max() - yimg.min()) * 255).byte().numpy()
        ax[0].imshow(yimg)
        #Image.fromarray(yimg).save('ranet_random.jpg')
        y = m2(x)[0]
        yimg = y[0].permute(1, 2, 0).cpu()
        yimg = ((yimg - yimg.min()) / (yimg.max() - yimg.min()) * 255).byte().numpy()
        ax[1].imshow(yimg)
        #Image.fromarray(yimg).save('vqvaegan_random.jpg')
        plt.show()
elif OPTION == 1:
    m = VQVAEGAN()
    m2 = RANet()
    sd = m.state_dict()
    print(len(m.state_dict().keys()))
    print('vqvaegan nparams:', sum(p.numel() for p in m.parameters()))
    print(len(m2.state_dict().keys()))
    print('ra nparams:', sum(p.numel() for p in m2.parameters()))
    m2.load_state_dict(sd, strict=False)
elif OPTION == 2: # test loading reference VQVAE from path, requires a pretrained ref. VQVAE in '/tmp/last.ckpt'
    m = RAModel(
        ddconfig={
            'target': 'RestoreFormer.modules.vqvae.vqvae_arch.RANet',
            'params': {
                'fix_decoder': False,
                'fix_codebook': False,
                'fix_encoder': False,
                'z_channels': 512,
            }
            },
        lossconfig={
            'target': 'RestoreFormer.modules.losses.vqperceptual.VQLPIPSWithDiscriminatorWithCompWithIdentity',
            'params': {
                'disc_start': 10001
                }
            }
    )
    m.init_ref_vqvae_from_ckpt('/tmp/last.ckpt')