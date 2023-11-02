import os
import cv2
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
import datasets
# from dataset import ImageRawDataset, PartialCompEvalDataset, PartialCompDataset
import inference as infer
import pdb


class Trainer_for_grap(object):

    def __init__(self, args):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        #########调节控制 boundary########################
        self.boundary_output = args.model['boundary_for_output']

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))

            # logger
            if args.trainer['tensorboard']:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(
                    args.exp_path))
            else:
                self.tb_logger = None
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))

        # create model 初始化模型  对应参数选择不同的网络模型
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True,boundary_out = self.boundary_output,
                    boundary_with_shared = args.model['boundary_shared'] )

        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_pretrain is not None), \
            "load_iter and load_pretrain are exclusive."

        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0

        self.curr_step = self.start_iter

        # lr scheduler & datasets            ######设置学习率和初始化dataset 以及 dataloader
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]  ####初始化训练集

        if not args.validate:  # train
            # 设置学习率以及相关的衰减系数
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],
                args.model['lr_mults'],
                args.model['lr'],
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)

            #train_dataset = trainval_class(args.data, 'train')
            #train_dataset = trainval_class('./data/Grap/', 'putao', is_train=True,for_maskdeOcc= True, resize= (320,320))
            train_dataset = trainval_class('./data/Grap/', 'putao', is_train=True,
                                           resize=(320, 320),boundary_output = self.boundary_output)
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset,
                args.model['total_iter'],
                args.data['batch_size'],
                last_iter=self.start_iter - 1)
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=args.data['workers'],
                                           pin_memory=False,
                                           sampler=train_sampler)


        val_class = datasets.__dict__[args.data['onlyVal_dataset']]
        val_dataset = val_class('./data/Grap/putao/real_inmodal/', resize= (320,320))
        val_sampler = utils.DistributedSequentialSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.data['batch_size_val'],
            shuffle=False,
            num_workers=args.data['workers'],
            pin_memory=False,
            sampler=val_sampler)

        self.args = args

    def run(self):

        # offline validate
        if self.args.validate:
            self.validate('off_val')
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')

        # train
        self.train()

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()
        for i, inputs in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            if self.boundary_output:
               Intact_image,Occder_image,Occ_image = inputs[0],inputs[1],inputs[2]
               b,c,h,w = Intact_image.shape
               rgb_image = torch.zeros((b,3,h,w),dtype=torch.int8)
               self.model.set_input(rgb= rgb_image,mask =Occ_image[:,:1,:,:],eraser =Occder_image[:,:1,:,:],target = Intact_image[:,:1,:,:],
                                 target_boundary= Intact_image[:,1:2,:,:], for_boundary= self.boundary_output)
               loss_dict = self.model.step()
            else :
                Intact_image, Occder_image, Occ_image = inputs[0], inputs[1], inputs[2]
                b, c, h, w = Intact_image.shape
                rgb_image = torch.zeros((b, 3, h, w), dtype=torch.int8)
                self.model.set_input(rgb=rgb_image, mask=Occ_image, eraser=Occder_image,
                                     target=Intact_image,
                                     target_boundary=None, for_boundary=False)
                loss_dict = self.model.step()


            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                'print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg,
                                                  self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step,
                                               len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                    (self.curr_step % self.args.trainer['save_freq'] == 0 or
                     self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.args.exp_path),
                    self.curr_step)
            ###########################################
            visual_for_train = self.args.trainer['visual_for_train']
            ##########################################
            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                    self.curr_step == self.args.model['total_iter']):
               if visual_for_train:
                tensor_dict = self.model.forward_only(ret_loss=False, visual_forrealdata=False)
                all_together = []
                all_together.append(
                    utils.visualize_tensor(tensor_dict,
                                           self.args.data.get('data_mean', [0, 0, 0]),
                                           self.args.data.get('data_std', [1, 1, 1])))
                all_together = torch.cat(all_together, dim=2)
                grid = vutils.make_grid(all_together,
                                        nrow=1,
                                        normalize=True,
                                        range=(0., 255.),
                                        scale_each=False)
                import matplotlib.pyplot as plt
                arrayImg = grid.numpy()
                arrayShow = np.squeeze(arrayImg[:1, :, :], 0)

                plt.figure(figsize=(192, 160))
                plt.imshow(arrayShow)
                plt.savefig("{}/images/OUTPUT_{}_{}.png".format(
                    self.args.exp_path, 'train', self.curr_step), dpi=40)
                plt.close()
                self.validate('on_val')
               else:
                self.validate('on_val')
    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        all_together = []
        for i, inputs in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and
                    self.args.trainer['val_iter'] != -1 and
                    i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)
            real_image1,real_image2 = inputs[0], inputs[1]
            b, c, h, w = real_image1.shape
            b,c,h,w = real_image1.shape
            rgb_image = torch.zeros((b,3,h,w),dtype=torch.int8)
            self.model.set_input(rgb= rgb_image,mask =real_image1,eraser =real_image2,target = None,
                                 target_boundary = None, for_boundary= self.boundary_output)
            tensor_dict = self.model.forward_only(ret_loss=False,visual_forrealdata = True)
            btime_rec.update(time.time() - end)
            end = time.time()

            # tb visualize
            if self.rank == 0:
                disp_start = max(self.args.trainer['val_disp_start_iter'], 0)                       #val_disp_start_iter = 0
                disp_end = min(self.args.trainer['val_disp_end_iter'], len(self.val_loader))        #val_disp_end_iter = 1
                if (i >= disp_start and i < disp_end):
                    all_together.append(
                        utils.visualize_tensor(tensor_dict,
                                               self.args.data.get('data_mean', [0, 0, 0]),
                                               self.args.data.get('data_std', [1, 1, 1])))
                if (i == disp_end - 1 and disp_end > disp_start):
                    all_together = torch.cat(all_together, dim=2)
                    grid = vutils.make_grid(all_together,
                                            nrow=1,
                                            normalize=True,
                                            range=(0., 255.),
                                            scale_each=False)
                    import matplotlib.pyplot as plt
                    arrayImg = grid.numpy()
                    arrayShow = np.squeeze(arrayImg[:1,:,:], 0)

                    plt.figure(figsize=(192, 160))
                    plt.imshow(arrayShow)
                    plt.savefig("{}/images/OUTPUT_{}_{}.png".format(
                        self.args.exp_path, phase, self.curr_step),dpi=40)
                    plt.close()
                    if self.tb_logger is not None:
                        self.tb_logger.add_image('Image_' + phase, grid,
                                                 self.curr_step)

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')

