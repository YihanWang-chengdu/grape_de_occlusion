import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss, SSIM_Loss,Weightedboundaryloss
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pdb

class PartialCompletionMask(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False,boundary_out= True,boundary_with_shared = False):
        #初始化模型及结构输出
        super(PartialCompletionMask, self).__init__(params, dist_model,boundary_out,boundary_with_shared = False )

        self.params = params
        self.use_rgb = params['use_rgb']
        self.mse = torch.nn.MSELoss(size_average= True)

        # loss
        self.criterion = MaskWeightedCrossEntropyLoss(
            inmask_weight=params['inmask_weight'],
            outmask_weight=1.)
        self.criterion_boundary = Weightedboundaryloss(
            inmask_weight = params['inmask_weight'],
            outmask_weight=1. )
        self.criterion_ssim = SSIM_Loss(
            window_size= 11,
            channel= 1,
            inmask_weight=params['inmask_weight'],
            outmask_weight=1.
        )
    def MSE_for_output(self,output,target,eraser,mask):
        n, c, h, w = output.size()
        comp = output.argmax(dim=1, keepdim=True).float()
        comp[eraser == 0] = (mask > 0).float()[eraser == 0]
        criterion = torch.nn.MSELoss(reduction='mean')

        eraser = eraser.bool().view(n,h,w)
        target_inmask = target[eraser]
        target_outmask = target[~eraser]

        predict = comp.transpose(1, 2).transpose(2, 3).contiguous()
        inmask_pre = predict[eraser.view(n, h, w, 1)].view(-1, 1)
        outmask_pre =predict[(~eraser).view(n, h, w, 1)].view(-1, 1)


        return criterion(inmask_pre,target_inmask) + criterion(outmask_pre,target_outmask)

    def dice_loss_func(self,input, target):
        smooth = 1.
        n = input.size(0)
        iflat = input.view(n, -1)
        tflat = target.view(n, -1)
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((2. * intersection + smooth) /
                    (iflat.sum(1) + tflat.sum(1) + smooth))
        return loss.mean()

    def boundary_loss_func(self,predict, gtmasks, erase):

        n, _, h, w = predict.shape
        erase = erase.bool()
        gtmasks_inmask = gtmasks[erase]
        gtmasks_outmask = gtmasks[~erase]

        predict_inmask = predict[erase]
        predict_outmask = predict[(~erase)]

        bce_loss_inmask = torch.nn.functional.binary_cross_entropy_with_logits(predict_inmask, gtmasks_inmask) / \
                          predict_inmask.shape[0]
        bce_loss_outmask = torch.nn.functional.binary_cross_entropy_with_logits(gtmasks_outmask, predict_outmask) / \
                           predict_outmask.shape[0]

        return bce_loss_inmask*self.params['inmask_weight'] + bce_loss_outmask

    def set_input(self, rgb=None, mask=None, eraser=None, target=None,target_boundary = None, for_boundary= False):
        if for_boundary:
           if target is None:                 #验证的时候，没有真实的标签可供读取
               if mask.shape[1] == 1:
                 self.rgb = rgb.cuda()
                 self.mask = mask.cuda()
                 self.eraser = eraser.cuda()
                 self.for_boundary = True
               else:
                raise ('输入的数据格式存在问题，通道数量应该为2,其中第二通道包含了边缘信息')
           else:
             if mask.shape[1]== 1:
              self.rgb = rgb.cuda()
              self.mask = mask.cuda()
              self.eraser = eraser.cuda()
              self.target = target.cuda()
              self.target_boundary = target_boundary.cuda()
              self.for_boundary = True
             else:
              raise ('输入的数据格式存在问题，通道数量应该为2,其中第二通道包含了边缘信息')
        else:
         if target is None:
             self.rgb = rgb.cuda()
             self.mask = mask.cuda()
             self.eraser = eraser.cuda()
             self.for_boundary = False
         else:
            if mask.shape[1] == 1 :
             self.rgb = rgb.cuda()
             self.mask = mask.cuda()
             self.eraser = eraser.cuda()
             self.target = target.cuda()
             self.for_boundary = False
            else:
             raise ('输入的数据格式存在问题，通道数量应该为1')
    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        order_method = self.params.get('order_method', 'ours')
        # order
        if order_method == 'ours':
            order_matrix = infer.infer_order2(
                self, image, inmodal, category, bboxes,
                use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_order'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_order', 0),
                input_size=input_size,
                min_input_size=16,
                interp=self.params['inference']['order_interp'])
        elif order_method == 'hull':
            order_matrix = infer.infer_order_hull(inmodal)
        elif order_method == 'area':
            order_matrix = infer.infer_order_area(inmodal, above=self.params['above'])
        elif order_method == 'yaxis':
            order_matrix = infer.infer_order_yaxis(inmodal)
        else:
            raise Exception("No such method: {}".format(order_method))

        gt_order_matrix = infer.infer_gt_order(inmodal, amodal)
        allpair_true, allpair, occpair_true, occpair, show_err = infer.eval_order(
            order_matrix, gt_order_matrix)

        # amodal
        amodal_method = self.params.get('amodal_method', 'ours')
        if amodal_method == 'ours':
            amodal_patches_pred = infer.infer_amodal(
                self, image, inmodal, category, bboxes,
                order_matrix, use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_amodal'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_amodal', 0),
                input_size=input_size,
                min_input_size=16, interp=self.params['inference']['amodal_interp'],
                order_grounded=self.params['inference']['order_grounded'])
            amodal_pred = infer.patch_to_fullimage(
                amodal_patches_pred, bboxes,
                image.shape[0], image.shape[1],
                interp=self.params['inference']['amodal_interp'])
        elif amodal_method == 'hull':
            amodal_pred = np.array(infer.infer_amodal_hull(
                inmodal, bboxes, order_matrix,
                order_grounded=self.params['inference']['order_grounded']))
        elif amodal_method == 'raw':
            amodal_pred = inmodal # evaluate raw
        else:
            raise Exception("No such method: {}".format(amodal_method))

        intersection = ((amodal_pred == 1) & (amodal == 1)).sum()
        union = ((amodal_pred == 1) | (amodal == 1)).sum()
        target = (amodal == 1).sum()

        return allpair_true, allpair, occpair_true, occpair, intersection, union, target

    def forward_only(self, ret_loss=True, visual_forrealdata = False):
        with torch.no_grad():
            if self.for_boundary:
              if self.use_rgb:
                 output,boundary = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
              else:
                 output,boundary = self.model(torch.cat([self.mask, self.eraser], dim=1))
              if output.shape[2] != self.mask.shape[2]:
                 output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
            else:
                if self.use_rgb:
                    output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
                else:
                    output = self.model(torch.cat([self.mask, self.eraser], dim=1))
                if output.shape[2] != self.mask.shape[2]:
                    output = nn.functional.interpolate(
                        output, size=self.mask.shape[2:4],
                        mode="bilinear", align_corners=True)

        if not visual_forrealdata:           ########### visual_for_Train #############
             comp =  torch.nn.functional.softmax(output,dim=1)
             comp = comp.argmax(dim=1, keepdim=True).float()
             comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]

             vis_combo = (self.mask > 0).float()
             vis_combo[self.eraser == 1] = 0.5

             vis_target = self.target.cpu().clone().float()
             if vis_target.max().item() == 255:
                vis_target[vis_target == 255] = 0.5

             if self.use_rgb:
               cm_tensors = [self.rgb]
             else:
               cm_tensors = []

             if self.for_boundary:
                 boundary = torch.sigmoid(boundary)
                 ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors': [self.mask, vis_combo, comp,boundary, vis_target,self.target_boundary]}
             else:
                 ret_tensors = {'common_tensors': cm_tensors,
                                'mask_tensors': [self.mask, vis_combo, comp, vis_target]}
             if ret_loss:
                  loss = self.criterion(output, self.target.long(), self.eraser.squeeze(1)) / self.world_size
                  return ret_tensors, {'loss': loss}
             else:
                  return ret_tensors
        else:         ###########################Validation ##########################
            comp = torch.nn.functional.softmax(output, dim=1)
            comp = comp.argmax(dim=1, keepdim=True).float()
            comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]

            vis_combo = (self.mask > 0).float()
            vis_combo[self.eraser == 1] = 0.5

            #############################反向输出一些可视化结果试试##############################
            if self.for_boundary:
                revise_output, revise_boundary = self.model(torch.cat([self.eraser, self.mask], dim=1))
            else:
                 revise_output = self.model(torch.cat([self.eraser, self.mask], dim=1))

            revise_comp = torch.nn.functional.softmax(revise_output, dim=1)
            revise_comp = revise_comp.argmax(dim=1, keepdim=True).float()
            revise_comp[self.mask == 0] = (self.eraser > 0).float()[self.mask == 0]

            revise_vis_combo = (self.eraser > 0).float()
            revise_vis_combo[self.mask == 1] = 0.5


            if self.use_rgb:
                cm_tensors = [self.rgb]
            else:
                cm_tensors = []

            if self.for_boundary:
               boundary = torch.sigmoid(boundary)
               ret_tensors = {'common_tensors': cm_tensors,
                           'mask_tensors': [self.mask, vis_combo, comp, boundary]}
            else:
               ret_tensors = {'common_tensors': cm_tensors,
                               'mask_tensors': [self.mask, vis_combo, comp]}
            ##################控制验证集上的输出是否同时输出反向验证结果############################
            visualization_revise_output  = True
            ################################################################################
            if visualization_revise_output:
                ret_tensors = {'common_tensors': cm_tensors,
                               'mask_tensors': [self.mask, vis_combo, comp,self.eraser,revise_vis_combo,revise_comp]}

            if ret_loss:
                loss = self.criterion(output, self.target.long(), self.eraser.squeeze(1)) / self.world_size
                return ret_tensors, {'loss': loss}
            else:
                return ret_tensors

    def step(self):
        if self.for_boundary:
            if self.use_rgb:
                output,boundary = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
            else:
                output,boundary = self.model(torch.cat([self.mask, self.eraser], dim=1))

            loss = self.criterion(output, self.target.long().squeeze(1), self.eraser.squeeze(1)) / self.world_size
            #########基于葡萄掩码全局的mse监督
            #loss_boundary = self.criterion_boundary(boundary, self.target, self.eraser)
            ##############基于葡萄边缘的bce监督
            loss_boundary = self.boundary_loss_func(boundary, self.target_boundary, self.eraser) / self.world_size
            loss = loss+loss_boundary
        else:
            if self.use_rgb:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
            else:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1))
        #############################################################
            loss = self.criterion(output, self.target.long().squeeze(1), self.eraser.squeeze(1)) / self.world_size
        #loss2=  self.MSE_for_output(output,self.target, self.eraser,self.mask)
        #loss2 = self.criterion_ssim(output, self.target.long(), self.eraser,self.mask) / self.world_size
        #loss=loss+loss2
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
