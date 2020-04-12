import utility
from types import SimpleNamespace

from model import common
from loss import discriminator_hair

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Adversarial_hair(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial_hair, self).__init__()
        self.gan_k = args.gan_k
        self.dis = discriminator_hair.Discriminator_hair(args)
        optim_dict = {
            'optimizer': 'ADAM',
            'betas': (0, 0.9),
            'epsilon': 1e-8,
            'lr': 1e-5,
            'weight_decay': args.weight_decay,
            'decay': args.decay,
            'gamma': args.gamma
        }
        optim_args = SimpleNamespace(**optim_dict)

        self.optimizer = utility.make_optimizer(optim_args, self.dis)

    def forward(self, outputs, targets, masks):
        fake = outputs[0]
        real = targets[0]
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach, masks)
            d_real = self.dis(real, masks)
            retain_graph = False
            
            #'GAN':
            # loss_d = self.bce(d_real, d_fake)
            
            #'WGAN'
            loss_d = (d_fake - d_real).mean()
            epsilon = torch.rand(fake.shape[0], 1, 1, 1)
            epsilon = epsilon.expand_as(real)
            epsilon = epsilon.cuda()

            hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
            hat.requires_grad = True
            d_hat = self.dis(hat, masks)
            gradients = torch.autograd.grad(
                outputs=d_hat.sum(), inputs=hat,
                retain_graph=True, create_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
            loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            for p in self.dis.parameters():
                p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake, masks)      # for backpropagation, use fake as it is
        
        #'GAN':
        # label_real = torch.ones_like(d_fake_bp)
        # loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        
        #'WGAN'
        loss_g = -d_fake_bp.mean()
        
        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
