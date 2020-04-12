import os
import math
from decimal import Decimal
import numpy as np
import cv2

import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        print('=> Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / (1024. * 1024)))


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, gallery_tensors, mask, subID, filename, _) in enumerate(self.loader_train):

            hr_p, lr_p_x2, lr_p_x4, lr_f_x2, lr_f_x4, hr_f, mask = self.prepare(lr[0], lr[1], lr[2], lr[3], lr[4], hr, mask)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            out = self.model([hr_p, lr_p_x2, lr_p_x4]) # I128_fake, I64_fake, I32_fake
            target = [hr_f, lr_f_x2, lr_f_x4]

            loss = self.loss(out, target, mask)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), 1)
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            top1 = utility.AverageMeter()
            top5 = utility.AverageMeter()
            for lr, hr, gallery_tensors, mask, subID, filename,_ in tqdm(d, ncols=80):
                hr_p, lr_p_x2, lr_p_x4, hr_f, gallery_tensors, subID = self.prepare(
                        lr[0], lr[1], lr[2], hr, gallery_tensors, subID)
                outputs = self.model([hr_p, lr_p_x2, lr_p_x4])

                sr = outputs[0]
                
                sr = utility.quantize(sr, self.args.rgb_range)
                
                self.ckp.log[-1, idx_data, 0] += utility.calc_psnr(
                    sr, hr_f, 1, self.args.rgb_range, dataset=d
                )

                # only save regular illum
                ind = []
                for i in range(sr.size()[0]):
                    illum = filename[i].split('_')[4]
                    if illum in ['09', '15']:
                        ind.append(i)

                if ind:
                    save_list = [sr]
                    if self.args.save_gt:
                        save_list.extend([hr_p, hr_f])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list)

            self.ckp.log[-1, idx_data, 0] /= len(d)
            best = self.ckp.log.max(0)
            print(d.dataset.name,
                    self.ckp.log[-1, idx_data, 0],
                    best[0][idx_data, 0],
                    best[1][idx_data, 0] + 1)
            self.ckp.write_log(
                '[{} ]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.ckp.log[-1, idx_data, 0],
                    best[0][idx_data, 0],
                    best[1][idx_data, 0] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
