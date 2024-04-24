import sys, os

root_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'RestoreFormer/modules/losses'))

from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from RestoreFormer.modules.vqvae.arcface_arch import ResNetArcFace
from torch.utils.data import DataLoader
from basicsr.data import build_dataset
from copy import deepcopy
from lpips import LPIPS
import torch.nn.functional as F
import numpy as np
import math, argparse, torch, glob, cv2, tqdm

def calculate_fid_folder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inception model
    inception = load_patched_inception_v3(device)

    # create dataset
    opt = {
        'name': 'SingleImageDataset',
        'type': 'SingleImageDataset',
        'dataroot_lq': args.folder,
        'io_backend': {'type': args.backend},
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    }
    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    args.num_sample = min(args.num_sample, len(dataset))
    total_batch = math.ceil(args.num_sample / args.batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device).numpy()
    features = features[:args.num_sample]
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(args.fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    return fid

def calculate_psnr_ssim_lpips_idd_folder(args):
    gt_names = sorted(glob.glob(os.path.join(args.gt_folder, '*')))

    perceptual_loss = LPIPS().eval().cuda()
    identity = ResNetArcFace(block = 'IRBlock', 
                                  layers = [2, 2, 2, 2],
                                  use_se = False).eval().cuda()
    identity_model_path = 'experiments/pretrained_models/arcface_resnet18.pth'
    sd = torch.load(identity_model_path, map_location="cpu")
    for k, v in deepcopy(sd).items():
        if k.startswith('module.'):
            sd[k[7:]] = v
            sd.pop(k)
    identity.load_state_dict(sd, strict=True)

    mean_psnr = mean_ssim = mean_lpips = mean_norm_lpips = mean_idd = 0.
    individual_metrics = {} # k: gt filename, v: (psnr, ssim, lpips, idd)

    for i in tqdm.tqdm(range(len(gt_names))):
        # load images
        gt_name = gt_names[i].split('/')[-1][:-4]
        img_name = os.path.join(args.folder,f'{gt_name}{args.post}.png')
        if not os.path.exists(img_name):
            print(img_name, 'does not exist')
            continue

        img = cv2.imread(img_name)
        img = torch.FloatTensor(img).cuda().permute(2,0,1).unsqueeze(0) / 255
        gt = cv2.imread(gt_names[i])
        gt = torch.FloatTensor(gt).cuda().permute(2,0,1).unsqueeze(0) / 255

        # psnr
        cur_psnr = calculate_psnr_pt(img, gt, 0).item()
        cur_ssim = calculate_ssim_pt(img, gt, 0).item()

        # lpips
        cur_lpips = perceptual_loss(img, gt)[0].item()
        norm_lpips = perceptual_loss(2 * img - 1, 2 * gt - 1)[0].item()

        # idd
        def gray_resize_for_identity(out, size=128):
            out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
            out_gray = out_gray.unsqueeze(1)
            out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
            return out_gray

        with torch.no_grad():
            identity_gt = identity(gray_resize_for_identity(gt))
            identity_out = identity(gray_resize_for_identity(img))
            identity_loss = ((identity_gt * identity_out).sum() / identity_gt.norm() / identity_out.norm()).acos()

        # aggregating metrics
        individual_metrics[gt_name] = (cur_psnr, cur_ssim, cur_lpips, norm_lpips, identity_loss)

        mean_psnr += cur_psnr / len(gt_names)
        mean_ssim += cur_ssim / len(gt_names)
        mean_lpips += cur_lpips / len(gt_names)
        mean_norm_lpips += norm_lpips / len(gt_names)
        mean_idd += identity_loss / len(gt_names)

    return (mean_psnr, mean_ssim, mean_lpips, mean_norm_lpips, mean_idd), individual_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Path to the folder of reconstructions.')
    parser.add_argument('--gt_folder', type=str, help='Path to the GT')
    parser.add_argument('--fid_stats', type=str, help='Path to the dataset fid statistics.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_sample', type=int, default=50000, help='Max number of samples for calculating FID (default: 50000)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataset (default: 4)')
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb (default: disk)')
    parser.add_argument('--save_name', type=str, default='niqe', help='File name for saving results (default: niqe)')
    parser.add_argument('--post', type=str, default='', help='A prefix to add to the end of the GT image to get the corresponding filename of the reconstruction image (default: empty string)')
    args = parser.parse_args()

    fid = psnr = ssim = lpips = lpips_norm = idd = -1
    try:
        fid = calculate_fid_folder(args)
    except Exception as e:
        print('FID failed to run with exception:', e)

    individual_metrics = {}
    try:
        (psnr, ssim, lpips, lpips_norm, idd), individual_metrics = calculate_psnr_ssim_lpips_idd_folder(args)
    except Exception as e:
        print('PSNR, SSIM, LPIPS, IDD failed to run with exception:', e)

    # print metrics
    #print('FID, PSNR, SSIM, LPIPS, LPIPS (normalized inputs), IDD')
    print(f'{fid},{psnr},{ssim},{lpips},{lpips_norm},{idd}')

    # write to file
    fout = open(f'{args.save_name}.csv', 'w')
    fout.write('GT name,PSNR,SSIM,LPIPS,LPIPS (norm),IDD\n')
    for gt_name, img_metrics in individual_metrics.items():
        value_str = ','.join(map(str, img_metrics))
        fout.write(f'{gt_name},{value_str}\n')
    fout.close()
    print(f'Finished writing to {args.save_name}.csv')