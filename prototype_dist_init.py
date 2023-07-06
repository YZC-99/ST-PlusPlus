import argparse
import logging
from collections import OrderedDict
import  time
import datetime
import os
import csv
from torchmetrics import Dice
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet

from torch.nn import CrossEntropyLoss, DataParallel

from core.configs import cfg
from core.datasets import build_dataset
# from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from utils import count_params, meanIOU

from core.utils.prototype_dist_estimator import prototype_dist_estimator

import warnings
warnings.filterwarnings('ignore')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def prototype_dist_init(cfg,src_train_loader):
    #初始化模型
    num_classes = 2
    ckpt_path = cfg.MODEL.stage1_ckpt_path
    print("计算prototype加载的模型是：{}".format(ckpt_path))

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo['deeplabv3plus']('resnet50',num_classes)

    sd = torch.load(ckpt_path,map_location='cpu')
    new_state_dict = {}
    for key, value in sd.items():
        if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
            if 'module.' + key in model.state_dict():
                # 模型在多GPU上训练并保存，加载权重时加上"module."前缀
                key = 'module.' + key
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    torch.cuda.empty_cache()



    iteration = 0

    model.eval()
    model.to("cuda")
    end = time.time()
    start_time = time.time()
    max_iters = len(src_train_loader)
    meters = MetricLogger(delimiter="  ")
    tbar = tqdm(src_train_loader)
    # 创建csv文件
    with open(os.path.join(cfg.prototype_path, 'prototype_metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'IoU', 'Dice'])  # 写入表头
        with torch.no_grad():
            dice = Dice(num_classes=cfg.MODEL.NUM_CLASSES, average='macro')
            metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)
            for i, (src_input, src_label) in enumerate(tbar):
                data_time = time.time() - end

                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long()

                # backbone得输出
                src_feat = model.backbone.base_forward(src_input)[-1]
                # src_feat = model(src_input)
                # 结果输出
                src_out = model(src_input)

                """
                由于不知道加载模型的权重是否正确，并且需要验证计算prototype的正确性，因此推理得到模型的输出，并计算指标            
                """
                pred = torch.argmax(src_out, dim=1)
                metric.add_batch(pred.detach().cpu().numpy(), src_label.cpu().numpy())
                dice_score = dice(pred.cpu(), src_label.cpu())
                mIOU = metric.evaluate()[-1]
                writer.writerow([i, mIOU, dice_score.item()])
                tbar.set_description('current_mIOU: %.2f' % (mIOU * 100.0))
                # src_feat = feature_extractor(src_input)
                # src_out = classifier(src_feat)


                B, N, Hs, Ws = src_feat.size()
                _, C, H, W = src_out.size()

                # source mask: downsample the ground-truth label
                src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
                src_mask = src_mask.contiguous().view(B * Hs * Ws, )

                # feature level
                src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
                feat_estimator.update(features=src_feat.detach().clone(), labels=src_mask)

                # output level
                src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
                src_out_mask = src_label.unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(B * H * W,)
                out_estimator.update(features=src_out.detach().clone(), labels=src_out_mask)

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)

                iteration = iteration + 1
                eta_seconds = meters.time.global_avg * (max_iters - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


                if iteration == max_iters:
                    feat_estimator.save(name='prototype_feat_dist.pth')
                    out_estimator.save(name='prototype_out_dist.pth')



