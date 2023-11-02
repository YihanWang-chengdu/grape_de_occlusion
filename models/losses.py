import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class L2LossWithIgnore(nn.Module):

    def __init__(self, ignore_value=None):
        super(L2LossWithIgnore, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target): # N1HW, N1HW
        if self.ignore_value is not None:
            target_area = target != self.ignore_value
            target = target.float()
            return (input[target_area] - target[target_area]).pow(2).mean()
        else:
            return (input - target.float()).pow(2).mean()
class Weightedboundaryloss(nn.Module):
    def __init__(self, inmask_weight=5, outmask_weight=1):
        super(Weightedboundaryloss, self).__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight
        self.mse = torch.nn.MSELoss(size_average=True)

    def forward(self, predict, target, mask):
        '''
        predict: N1HW
        target: N1HW
        mask: N1HW        #eraser
        '''
        predict = torch.nn.functional.sigmoid(predict)
        n, _, h, w = predict.size()
        # mask = mask.byte()
        mask = mask.bool()
        target_inmask = target[mask]
        target_outmask = target[~mask]

        predict_inmask = predict[mask]
        predict_outmask = predict[(~mask)]

        loss_inmask = self.mse(predict_inmask, target_inmask)
        loss_outmask = self.mse(predict_outmask, target_outmask)

        loss = (self.inmask_weight * loss_inmask + self.outmask_weight * loss_outmask)
        return loss

class MaskWeightedCrossEntropyLoss(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super(MaskWeightedCrossEntropyLoss, self).__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight
        self.mse = torch.nn.MSELoss( size_average=True)

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW        #eraser
        '''
        predict = torch.nn.functional.softmax(predict,dim=1)
        n, c, h, w = predict.size()
        #mask = mask.byte()
        mask = mask.bool()
        target_inmask = target[mask]
        target_outmask = target[~mask]

        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict_inmask = predict[mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        predict_outmask = predict[(~mask).view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss_inmask = nn.functional.cross_entropy(
            predict_inmask, target_inmask, size_average=False)
        loss_outmask = nn.functional.cross_entropy(
            predict_outmask, target_outmask, size_average=False)

        loss = (self.inmask_weight * loss_inmask + self.outmask_weight * loss_outmask) / (n * h * w)
        return loss

from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

from functools import partial
import torch.nn.functional as F
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True, inmask_weight=5, outmask_weight=1):
        super().__init__()
        window = create_window(window_size, channel)
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight
        self.ssim = partial(_ssim,
                            window=window.cuda(),
                            window_size=window_size,
                            channel=channel,
                            size_average=size_average)

    def forward(self, predict, target, eraser,mask):

        n, c, h, w = predict.size()
        # mask = mask.bool()
        # target_inmask = target[mask]
        # target_outmask = target[~mask]
        # predict = predict.contiguous()
        #
        # predict_inmask = predict[mask.view(n,1, h, w).repeat(1, c, 1, 1)][:,:1,:,:]
        # predict_outmask = predict[(~mask).view(n,1, h, w).repeat(1, c, 1, 1)][:,:1,:,:]

        comp = predict.argmax(dim=1, keepdim=True).float()
        comp[eraser == 0] = (mask > 0).float()[eraser == 0]


        ssim_loss = 1 - self.ssim(comp[:,:1,:,:].float(), target.view(n,-1,h,w).float())
        return ssim_loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict
