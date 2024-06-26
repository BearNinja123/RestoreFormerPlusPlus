from torchvision.transforms.functional import (
        adjust_brightness, adjust_contrast,
        adjust_hue, adjust_saturation, normalize)
from basicsr.utils import (
        FileClient, get_root_logger, imfrombytes,
        img2tensor, imwrite, tensor2img)
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment
from omegaconf import OmegaConf
from glob import glob

import torch.utils.data as data
import os.path as osp
import numpy as np
import torch, cv2
import math, os, random, argparse, pdb

@DATASET_REGISTRY.register()
class RefDataset(data.Dataset):
    def __init__(self, opt):
        super(RefDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

        if self.crop_components:
            self.components_list = torch.load(opt.get('component_path'))

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError()
            #self.io_backend_opt['db_paths'] = self.gt_folder
            #if not self.gt_folder.endswith('.lmdb'):
            #    raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            #with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
            #    self.paths = [line.split('.')[0] for line in fin]
        else:
            #self.paths = paths_from_folder(self.gt_folder)
            self.identity_folders = []
            for identity in glob(f'{self.gt_folder}/*'):
                if len(glob(f'{identity}/*')) > 1:
                    self.identity_folders.append(identity)
            # self.paths.sort()

        # degradations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, '
                        f'shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')

        self.color_jitter_shift /= 255.

        self.get_mask = opt.get('get_mask', False)
        self.mask_root = opt.get('mask_root')

        self.uneven_prob = opt.get('uneven_prob', 0.)
        self.hazy_prob = opt.get('hazy_prob', 0.)
        self.hazy_alpha = opt['hazy_alpha']

        self.exposure_prob = opt.get('exposure_prob', 0.)
        self.exposure_range = opt['exposure_range']

        self.shift_prob = opt.get('shift_prob', 0.)
        self.shift_unit = opt.get('shift_unit', 32)
        self.shift_max_num = opt.get('shift_max_num', 3)
        self.img_size = opt.get('img_size', (512, 512))

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        identity = self.identity_folders[index]
        identity_imgs = glob(f'{identity}/*')
        gt_path, ref_path = random.sample(identity_imgs, 2)
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(ref_path)
        img_ref = imfrombytes(img_bytes, float32=True)
        if img_ref.shape != self.img_size:
            img_ref = cv2.resize(img_ref, self.img_size)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        img_ref, status = augment(img_ref, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        if img_gt.shape != self.img_size:
            img_gt = cv2.resize(img_gt, self.img_size)
        h, w, _ = img_gt.shape

        if self.get_mask:
            img_name = gt_path.split('/')[-1]
            mask_path = os.path.join(self.mask_root, img_name)

            mask_bytes = self.file_client.get(mask_path)
            mask = imfrombytes(mask_bytes, flag='grayscale', float32=True)
            mask = np.expand_dims(mask, axis=-1)
            if status[0]:
                cv2.flip(mask, 1, mask)

            # mask = img2tensor(mask, bgr2rgb=True, float32=True)

        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        if (self.exposure_prob is not None) and (np.random.uniform() < self.exposure_prob):
                exp_scale = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
                img_gt *= exp_scale
        if (self.exposure_prob is not None) and (np.random.uniform() < self.exposure_prob):
                exp_scale = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
                img_ref *= exp_scale

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob ): # augment gt img
            # self.shift_unit = 32
            # import pdb
            # pdb.set_trace()
            shift_vertical_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), 
                                mode='symmetric')
            img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

            if self.crop_components:
                loc_left_eye += self.shift_max_num * self.shift_unit
                loc_left_eye[0] -= shift_h
                loc_left_eye[1] -= shift_v
                loc_left_eye[2] -= shift_h
                loc_left_eye[3] -= shift_v

                loc_right_eye += self.shift_max_num * self.shift_unit
                loc_right_eye[0] -= shift_h
                loc_right_eye[1] -= shift_v
                loc_right_eye[2] -= shift_h
                loc_right_eye[3] -= shift_v

                loc_mouth += self.shift_max_num * self.shift_unit
                loc_mouth[0] -= shift_h
                loc_mouth[1] -= shift_v
                loc_mouth[2] -= shift_h
                loc_mouth[3] -= shift_v

            if self.get_mask:
                mask_pad = np.pad(mask, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), 
                                mode='symmetric')
                mask = mask_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob ): # same for ref
            shift_vertical_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_ref_pad = np.pad(img_ref, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), 
                                mode='symmetric')
            img_ref = img_ref_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

        # ------------------------ generate lq image ------------------------ #
        img_lqs = []

        uneven = np.random.uniform()
        for i in range(2):

            # blur
            assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
            cur_kernel_size = random.randint(self.blur_kernel_size[0],self.blur_kernel_size[1]) * 2 + 1
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                cur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
            img_lq = cv2.filter2D(img_gt, -1, kernel)

            ## add simple hazy
            if (self.hazy_prob is not None) and (np.random.uniform() < self.hazy_prob):
                alpha = np.random.uniform(self.hazy_alpha[0], self.hazy_alpha[1])
                img_lq = img_lq * alpha + np.ones_like(img_lq) * (1 - alpha)

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            # noise
            if self.noise_range is not None:
                img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
            # jpeg compression
            if self.jpeg_range is not None:
                img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
            img_lqs.append(img_lq)

            if uneven >= self.uneven_prob:
                break

        if uneven >= self.uneven_prob:
            img_lq = img_lqs[0]
        else:
            ## crop out 64x64 patch from 512x512 for different degraded
            uneven_mask = np.zeros_like(img_gt)
            # patch_size = 128
            patch_size = random.randint(128,256)
            hs = random.randint(0, h-patch_size)
            ws = random.randint(0, w-patch_size)
            uneven_mask[hs:hs+patch_size, ws:ws+patch_size, :] = 1.0

            img_lq = img_lqs[0] * uneven_mask + img_lqs[1] * (1.0 - uneven_mask)


        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_ref, img_lq = img2tensor([img_gt, img_ref, img_lq], bgr2rgb=True, float32=True)
        if self.get_mask:
            mask = img2tensor(mask, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_ref, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'ref': img_ref,
                'gt_path': gt_path,
                'ref_path': ref_path,
            }
        if self.crop_components:
            return_dict['loc_left_eye'] = loc_left_eye
            return_dict['loc_right_eye'] = loc_right_eye
            return_dict['loc_mouth'] = loc_mouth
        if self.get_mask:
            return_dict['mask'] = mask

        return return_dict

    def __len__(self):
        #return len(self.paths)
        return len(self.identity_folders)

if __name__=='__main__':
    # pdb.set_trace()
    #base='configs/journal/RF18_mh1_new_params_multiscale_restart_v1_tc.yaml'
    base='configs/RestoreFormerPlusPlus.yaml'

    opt = OmegaConf.load(base)
    dataset = RefDataset(opt['data']['params']['train']['params'])

    for i in range(8):
        sample = dataset[i]
        name = sample['gt_path'].split('/')[-1][:-4]
        gt = tensor2img(sample['gt'], min_max=[-1, 1])
        ref = tensor2img(sample['ref'], min_max=[-1, 1])
        lq = tensor2img(sample['lq'], min_max=[-1, 1])
        
        imwrite(gt, name+'_gt.png')
        imwrite(ref, name+'_ref.png')
        imwrite(lq, name+'_lq_nojitter.png')
