from RestoreFormer.modules.vqvae.utils import get_roi_regions
from RestoreFormer.modules.vqvae.vqvae_arch import MEM_FMT
from main import instantiate_from_config
import torch.nn.functional as F
import pytorch_lightning as pl
import torch, warnings, time, math
warnings.filterwarnings('ignore')

def count_params(module):
    return sum(p.numel() for p in module.parameters())

class RestoreFormerModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000]
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig).to(memory_format=MEM_FMT)

        lossconfig['params']['distill_param'] = ddconfig['params']
        self.loss = instantiate_from_config(lossconfig).to(memory_format=MEM_FMT)
        self.loss.max_discriminator_weight = count_params(self.loss.discriminator) / count_params(self.vqvae)
        print("Loss max discriminator weight:", self.loss.max_discriminator_weight)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']
        
        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        for k in require_keys:
            if k not in keys: 
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd[k]

        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        self.load_state_dict(state_dict, strict=True)
        print(f"Restored from {path}")

    def forward(self, input):
        dec, diff, info, hs = self.vqvae(input)
        return dec, diff, info, hs

    @property
    def global_step(self): # needed since pytorch lightning increments global step with every optimizer step instead of every call to training_step 
        gs = super().global_step
        if gs >= 2 * self.disc_start and self.use_facial_disc:
            return (gs + 3 * self.disc_start) // 5
        return gs // 2

    '''
    PL 2.1.2
    Training with multiple optimizers is only supported with manual optimization.
    Remove the `optimizer_idx` argument from `training_step`, set `self.automatic_optimization = False`
    and access your optimizers in `training_step` with `opt1, opt2, ... = self.optimizers()`.
    '''
    def training_step(self, batch, batch_idx):
        #x = (batch[self.image_key] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
        is_power_2 = lambda x: int(math.log2(x)) == math.log2(x)
        if batch_idx == 0:
            self.tic = time.time()
        if self.global_rank == 0 and batch_idx > 16 and is_power_2(batch_idx + 1): # 16 is arbitrary, checking if batchidx +/- 1 is power of 2 to make sure logging images for visual inspection isn't included in the step speed
            print('it/s', (batch_idx / 2 - 2)/(time.time() - self.tic))
        if batch_idx > 16 and is_power_2(batch_idx - 1):
            self.tic = time.time()
        x = batch[self.image_key]
        xrec, qloss, info, hs = self(x)

        if self.image_key != 'gt':
            #x = (batch['gt'].to(self.device) - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt'].to(self.device)

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        opts = self.optimizers()
        loss, d_loss, d_left_eye_loss, d_right_eye_loss, d_mouth_loss, log_dict = self.loss(qloss, x, xrec, components, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # generator
        opts[0].zero_grad()
        self.manual_backward(loss)

        # discriminator
        opts[1].zero_grad()
        self.manual_backward(d_loss)

        #@torch.compile # compiling isn't any faster but keeping it here to remind myself that it won't help
        def fn():
            opts[0].step() 
            opts[1].step() 
        fn()
        
        if self.global_step >= self.disc_start and self.use_facial_disc:
            # left eye, right eye, and mouth loss
            for opt, loss_component in zip(opts[2:5], (d_left_eye_loss, d_right_eye_loss, d_mouth_loss)):
                opt.zero_grad()
                self.manual_backward(loss_component)
                opt.step() 

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        for s in self.lr_schedulers():
            s.verbose = False
            s.step()

    def validation_step(self, batch, batch_idx):
        #x = (batch[self.image_key] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
        x = batch[self.image_key]
        xrec, qloss, _info, _hs = self(x)

        if self.image_key != 'gt':
            #x = (batch['gt'] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt']

        aeloss, _discloss, _d_left_eye_loss, _d_right_eye_loss, _d_mouth_loss, log_dict_ae = self.loss(qloss, x, xrec, None, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        normal_params = []
        special_params = []
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            if 'decoder' in name and 'attn' in name:
                special_params.append(param)
            else:
                normal_params.append(param)
        opt_ae_params = [{'params': normal_params, 'lr': lr},
                         {'params': special_params, 'lr': lr*self.special_params_lr_scale}]
        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9), fused=True)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9), fused=True)

        optimizations = [opt_ae, opt_disc]

        s0 = torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=self.schedule_step, gamma=0.1, verbose=False)
        s1 = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=self.schedule_step, gamma=0.1, verbose=False)
        schedules = [s0, s1]

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]
            
            s2 = torch.optim.lr_scheduler.MultiStepLR(opt_l, milestones=self.schedule_step, gamma=0.1, verbose=False)
            s3 = torch.optim.lr_scheduler.MultiStepLR(opt_r, milestones=self.schedule_step, gamma=0.1, verbose=False)
            s4 = torch.optim.lr_scheduler.MultiStepLR(opt_m, milestones=self.schedule_step, gamma=0.1, verbose=False)
            schedules += [s2, s3, s4]

        return optimizations, schedules

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv #.weight
        return self.vqvae.decoder.conv_out #.weight

    @torch.no_grad
    def log_images(self, batch, **kwargs):
        self.vqvae.eval()
        log = dict()
        #x = (batch[self.image_key].to(self.device) - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
        x = batch[self.image_key].to(self.device)
        xrec, _qloss, _info, _hs = self(x)
        self.vqvae.train()
        log["inputs"] = x
        log["reconstructions"] = xrec

        if self.image_key != 'gt':
            #x = (batch['gt'] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt']
            log["gt"] = x
        return log

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if MEM_FMT != torch.contiguous_format:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.ndim == 4:
                    batch[k] = v.to(memory_format=MEM_FMT)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

class RBAModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000]
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig).to(memory_format=MEM_FMT)

        lossconfig['params']['distill_param'] = ddconfig['params']
        self.loss = instantiate_from_config(lossconfig).to(memory_format=MEM_FMT)
        self.loss.max_discriminator_weight = count_params(self.loss.discriminator) / count_params(self.vqvae)
        print("Loss max discriminator weight:", self.loss.max_discriminator_weight)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']
        
        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        state_dict = self.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        un_pretrained_keys = []
        for k in require_keys:
            if k not in keys: 
                # miss 'vqvae.'
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                state_dict[k] = sd[k]

        # print(f'*************************************************')
        # print(f"Layers without pretraining: {un_pretrained_keys}")
        # print(f'*************************************************')

        #self.load_state_dict(state_dict, strict=True)
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def forward(self, input, ref):
        dec, diff, info, hs = self.vqvae(input, ref)
        return dec, diff, info, hs

    @property
    def global_step(self): # needed since pytorch lightning increments global step with every optimizer step instead of every call to training_step 
        gs = super().global_step
        if gs >= 2 * self.disc_start and self.use_facial_disc:
            return (gs + 3 * self.disc_start) // 5
        return gs // 2

    def training_step(self, batch, batch_idx):
        # calculating training speed 
        is_power_2 = lambda x: int(math.log2(x)) == math.log2(x)
        if batch_idx == 0:
            self.tic = time.time()
        if self.global_rank == 0 and batch_idx > 16 and is_power_2(batch_idx + 1): # 16 is arbitrary, checking if batchidx +/- 1 is power of 2 to make sure logging images for visual inspection isn't included in the step speed
            print('it/s', (batch_idx / 2 - 2)/(time.time() - self.tic))
        if batch_idx > 16 and is_power_2(batch_idx - 1):
            self.tic = time.time()

        x = batch[self.image_key]
        ref = batch['ref'].to(self.device)
        xrec, qloss, info, hs = self(x, ref)

        if self.image_key != 'gt':
            #x = (batch['gt'].to(self.device) - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt'].to(self.device)

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        opts = self.optimizers()
        loss, d_loss, d_left_eye_loss, d_right_eye_loss, d_mouth_loss, log_dict = self.loss(qloss, x, xrec, components, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # generator
        opts[0].zero_grad()
        self.manual_backward(loss)
        opts[0].step() 

        # discriminator
        opts[1].zero_grad()
        self.manual_backward(d_loss)
        opts[1].step() 
        
        if self.global_step >= self.disc_start and self.use_facial_disc:
            # left eye, right eye, and mouth loss
            for opt, loss_component in zip(opts[2:5], (d_left_eye_loss, d_right_eye_loss, d_mouth_loss)):
                opt.zero_grad()
                self.manual_backward(loss_component)
                opt.step() 

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        for s in self.lr_schedulers():
            s.verbose = False
            s.step()

    def validation_step(self, batch, batch_idx):
        #x = (batch[self.image_key] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
        x = batch[self.image_key].to(self.device)
        ref = batch['ref'].to(self.device)
        xrec, qloss, _info, _hs = self(x, ref)

        if self.image_key != 'gt':
            #x = (batch['gt'] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt']

        aeloss, _discloss, _d_left_eye_loss, _d_right_eye_loss, _d_mouth_loss, log_dict_ae = self.loss(qloss, x, xrec, None, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        normal_params = []
        special_params = []
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue
            if 'decoder' in name and 'attn' in name:
                special_params.append(param)
            else:
                normal_params.append(param)
        opt_ae_params = [{'params': normal_params, 'lr': lr},
                         {'params': special_params, 'lr': lr*self.special_params_lr_scale}]
        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9), fused=True)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9), fused=True)

        optimizations = [opt_ae, opt_disc]

        s0 = torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=self.schedule_step, gamma=0.1, verbose=False)
        s1 = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=self.schedule_step, gamma=0.1, verbose=False)
        schedules = [s0, s1]

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]
            
            s2 = torch.optim.lr_scheduler.MultiStepLR(opt_l, milestones=self.schedule_step, gamma=0.1, verbose=False)
            s3 = torch.optim.lr_scheduler.MultiStepLR(opt_r, milestones=self.schedule_step, gamma=0.1, verbose=False)
            s4 = torch.optim.lr_scheduler.MultiStepLR(opt_m, milestones=self.schedule_step, gamma=0.1, verbose=False)
            schedules += [s2, s3, s4]

        return optimizations, schedules

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv
        return self.vqvae.decoder.conv_out

    @torch.no_grad
    def log_images(self, batch, **kwargs):
        self.vqvae.eval()
        log = dict()
        x = batch[self.image_key].to(self.device)
        ref = batch['ref'].to(self.device)
        xrec, _qloss, _info, _hs = self(x, ref)
        self.vqvae.train()
        log["inputs"] = x
        log["references"] = ref
        log["reconstructions"] = xrec

        if self.image_key != 'gt':
            #x = (batch['gt'] - batch['mean'][:, :, None, None]) * batch['rstd'][:, :, None, None]
            x = batch['gt']
            log["gt"] = x
        return log

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if MEM_FMT != torch.contiguous_format:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.ndim == 4:
                    batch[k] = v.to(memory_format=MEM_FMT)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
