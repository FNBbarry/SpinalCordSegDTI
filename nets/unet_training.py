import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def put_heatmap(shape, centers):
    axis_0_h = torch.max(centers[:, 0]) - torch.min(centers[:, 0])
    axis_1_w = torch.max(centers[:, 1]) - torch.min(centers[:, 1])
    sigma = max(axis_0_h, axis_1_w) / math.sqrt(4.6052 * 2)
    grid_x = torch.tile(torch.arange(shape[2]).unsqueeze(1).unsqueeze(2), (1,shape[3],centers.shape[0]))
    # np.repeat(np.expand_dims(np.tile(np.arange(shape[1]), (shape[0],1)),2),2,axis=2)
    grid_y = torch.tile(torch.arange(shape[3]).unsqueeze(0).unsqueeze(2), (shape[2],1,centers.shape[0]))
    grid_distance = (torch.square(grid_x.cuda()-centers[:,0].unsqueeze(0).unsqueeze(0)) + torch.square(grid_y.cuda() - centers[:,1].unsqueeze(0).unsqueeze(0)))/ 2.0 / sigma / sigma 
    grid_distance = torch.max(grid_distance,2)
    gaussian_heatmap = torch.exp(-grid_distance[0])
    # create a image based gaussian distribution
    for ind_x,ind_y in centers:
        gaussian_heatmap[ind_x,ind_y] = 1
    return gaussian_heatmap

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    # nt, ht, wt = target.size()
    # if h != ht and w != wt:
    #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    #
    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # temp_target = target.view(-1)
    guss_heatmap = torch.zeros_like(target)
    for i in range(n):
        for j in range(c):
            points = torch.stack(torch.where(target[i,j,:,:]==1),axis=1)
            if min(points.shape) == 0:
                continue
            guss_heatmap[i,j,:,:] = put_heatmap(target.shape, points)
    temp_inputs = inputs.view(c,-1)
    temp_target = guss_heatmap.view(c,-1)

    # CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    # return CE_loss
    dist_loss = torch.cdist(temp_inputs,temp_target,p=2)
    return dist_loss.diagonal().mean()

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)#total_iters=UnFreeze_Epoch
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        # the function of partial is to freeze some paramters of a function,and return a new function
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
