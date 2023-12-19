from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
#from basicsr.losses.losses import GANLoss, L1Loss
from basicsr.losses.basic_loss import L1Loss
from basicsr.losses.gan_loss import GANLoss

from RestoreFormer.modules.discriminator.model import (NLayerDiscriminator,
                                                       weights_init)
from RestoreFormer.modules.losses.lpips import LPIPS
from RestoreFormer.modules.vqvae.arcface_arch import ResNetArcFace
from RestoreFormer.modules.vqvae.facial_component_discriminator import \
    FacialComponentDiscriminator


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQLPIPSWithDiscriminatorWithCompWithIdentity(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 max_discriminator_weight=1e4, perceptual_weight=1.0, use_actnorm=False, 
                 disc_ndf=64, disc_loss="hinge", comp_weight=0.0, comp_style_weight=0.0, 
                 identity_weight=0.0, comp_disc_loss='vanilla', lpips_style_weight=0.0,
                 identity_model_path=None, **ignore_kwargs):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(style_weight=lpips_style_weight).to(memory_format=torch.contiguous_format).eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init).to(memory_format=torch.contiguous_format)
        if comp_weight > 0:
            self.net_d_left_eye = FacialComponentDiscriminator().to(memory_format=torch.contiguous_format)
            self.net_d_right_eye = FacialComponentDiscriminator().to(memory_format=torch.contiguous_format)
            self.net_d_mouth = FacialComponentDiscriminator().to(memory_format=torch.contiguous_format)
            print(f'Use components discrimination')

            self.cri_component = GANLoss(gan_type=comp_disc_loss, 
                                         real_label_val=1.0, 
                                         fake_label_val=0.0, 
                                         loss_weight=comp_weight)

            if comp_style_weight > 0.:
                self.cri_style = L1Loss(loss_weight=comp_style_weight, reduction='mean')

        if identity_weight > 0:
            self.identity = ResNetArcFace(block = 'IRBlock', 
                                          layers = [2, 2, 2, 2],
                                          use_se = False).to(memory_format=torch.contiguous_format)
            print(f'Use identity loss')
            if identity_model_path is not None:
                sd = torch.load(identity_model_path, map_location="cpu")
                for k, v in deepcopy(sd).items():
                    if k.startswith('module.'):
                        sd[k[7:]] = v
                        sd.pop(k)
                self.identity.load_state_dict(sd, strict=True)

            for param in self.identity.parameters():
                param.requires_grad = False

            self.cri_identity = L1Loss(loss_weight=identity_weight, reduction='mean')


        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminatorWithCompWithIdentity running with {disc_loss} loss, d_factor: {disc_factor}, d_weight: {disc_weight}")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.max_discriminator_weight = max_discriminator_weight
        self.comp_weight = comp_weight
        self.comp_style_weight = comp_style_weight
        self.identity_weight = identity_weight
        self.lpips_style_weight = lpips_style_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is None:
            last_layer = self.last_layer[0]
        nll_grads = torch.autograd.grad(nll_loss, last_layer.weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer.weight, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, h * w)
        features_t = features.transpose(1, 2).contiguous() # fix stride warning
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def forward(self, codebook_loss, gts, reconstructions, components,
                global_step, last_layer=None, penult=None, split="train"):
        # now the GAN part
        rec_loss = (torch.abs(gts.contiguous(memory_format=torch.contiguous_format) - reconstructions.contiguous(memory_format=torch.contiguous_format))) * self.pixel_weight
        if self.perceptual_weight > 0:
            p_loss, p_style_loss = self.perceptual_loss(gts.contiguous(memory_format=torch.contiguous_format), reconstructions.contiguous(memory_format=torch.contiguous_format))
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
            p_style_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
    
        # generator update
        logits_fake = self.discriminator(reconstructions.contiguous(memory_format=torch.contiguous_format))
        g_loss = -torch.mean(logits_fake)

        try:
            if penult is None:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            else:
                nll_grads = torch.autograd.grad(nll_loss, last_layer.weight, retain_graph=True)[0]

                reconstructions_adv = last_layer(penult.detach())
                logits_fake_tmp = self.discriminator(reconstructions_adv.contiguous(memory_format=torch.contiguous_format))
                g_loss_tmp = -torch.mean(logits_fake_tmp)
                g_grads = torch.autograd.grad(g_loss_tmp, last_layer.weight, retain_graph=False)[0]

                d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                d_weight *= self.discriminator_weight
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        loss = nll_loss + d_weight.clamp(0, self.max_discriminator_weight) * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + p_style_loss

        log = {
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/p_style_loss": p_style_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean(),
                }

        d_left_eye_loss = d_right_eye_loss = d_mouth_loss = 0.0
        if self.comp_weight > 0. and components is not None and self.discriminator_iter_start < global_step:
            # left eye disc
            fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(components['left_eyes'], return_feats=True)
            g_left_loss = self.cri_component(fake_left_eye, True, is_disc=False)

            real_left_eye, _real_left_eye_feats = self.net_d_left_eye(components['left_eyes_gt'])
            fake_left_eye, _fake_left_eye_feats = self.net_d_left_eye(components['left_eyes'].detach(), return_feats=True)
            d_left_eye_loss = self.cri_component(real_left_eye, True, is_disc=True) + self.cri_component(fake_left_eye, False, is_disc=True)

            # right eye disc
            fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(components['right_eyes'], return_feats=True)
            g_right_loss = self.cri_component(fake_right_eye, True, is_disc=False)

            real_right_eye, _real_right_eye_feats = self.net_d_right_eye(components['right_eyes_gt'])
            fake_right_eye, _fake_right_eye_feats = self.net_d_right_eye(components['right_eyes'].detach(), return_feats=True)
            d_right_eye_loss = self.cri_component(real_right_eye, True, is_disc=True) + self.cri_component(fake_right_eye, False, is_disc=True)

            # mouth disc
            fake_mouth, fake_mouth_feats = self.net_d_mouth(components['mouths'], return_feats=True)
            g_mouth_loss = self.cri_component(fake_mouth, True, is_disc=False)

            real_mouth, _real_mouth_feats = self.net_d_mouth(components['mouths_gt'])
            fake_mouth, _fake_mouth_feats = self.net_d_mouth(components['mouths'].detach(), return_feats=True)
            d_mouth_loss = self.cri_component(real_mouth, True, is_disc=True) + self.cri_component(fake_mouth, False, is_disc=True)
            
            loss += g_left_loss + g_right_loss + g_mouth_loss

            log.update({
                f"{split}/g_left_loss": g_left_loss.detach(),
                f"{split}/d_left_loss": d_loss.clone().detach().mean(),
                f"{split}/d_right_loss": d_loss.clone().detach().mean(),
                f"{split}/g_right_loss": g_right_loss.detach(),
                f"{split}/d_mouth_loss": d_loss.clone().detach().mean(),
                f"{split}/g_mouth_loss": g_mouth_loss.detach(),
            })

            if self.comp_style_weight > 0.:
                def _comp_style(feat, feat_gt, criterion):
                    return criterion(self._gram_mat(feat[0]), self._gram_mat(
                        feat_gt[0].detach())) * 0.5 + criterion(self._gram_mat(
                        feat[1]), self._gram_mat(feat_gt[1].detach()))

                _real_left_eye, real_left_eye_feats = self.net_d_left_eye(components['left_eyes_gt'])
                _real_right_eye, real_right_eye_feats = self.net_d_right_eye(components['right_eyes_gt'])
                _real_mouth, real_mouth_feats = self.net_d_mouth(components['mouths_gt'])

                comp_style_loss = 0.
                comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_style)
                comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_style)
                comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_style)
                loss = loss + comp_style_loss 
                log["{}/comp_style_loss".format(split)] = comp_style_loss.detach()

        if self.identity_weight > 0. and self.discriminator_iter_start < global_step:
            self.identity.eval()
            out_gray = self.gray_resize_for_identity(reconstructions)
            gt_gray = self.gray_resize_for_identity(gts)
            
            identity_gt = self.identity(gt_gray).detach()
            identity_out = self.identity(out_gray)

            identity_loss = self.cri_identity(identity_out, identity_gt)
            loss = loss + identity_loss 
            log["{}/identity_loss".format(split)] = identity_loss.detach()

        log["{}/total_loss".format(split)] = loss.clone().detach().mean()

        # second pass for discriminator update
        logits_fake = self.discriminator(reconstructions.contiguous(memory_format=torch.contiguous_format).detach())
        logits_real = self.discriminator(gts.contiguous(memory_format=torch.contiguous_format).detach())
        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        log.update({"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
                })

        return loss, d_loss, d_left_eye_loss, d_right_eye_loss, d_mouth_loss, log