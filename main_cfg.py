from dataset.my_semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
from torch.utils.tensorboard import SummaryWriter
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image

import torch
from torch.nn import CrossEntropyLoss, DataParallel
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from prototype_dist_init import prototype_dist_init

from core.configs import cfg
from yacs.config import CfgNode as CN
import sys
import yaml

from core.utils.train_methods import src_tgt_train,train,label

MODE = None

def load_model_ckpt(model,ckpt_path):
    sd = torch.load(ckpt_path,map_location='cpu')
    new_state_dict = {}
    for key, value in sd.items():
        if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
            if 'module.' + key in model.state_dict():
                # 模型在多GPU上训练并保存，加载权重时加上"module."前缀
                key = 'module.' + key
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')
    parser.add_argument("--config-file",
                        default="configs/sup.yaml",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    args = parser.parse_args()
    return args

def write_config(cfg, file):
    for key, value in cfg.items():
        if isinstance(value, CN):
            file.write(f"[{key}]\n")
            write_config(value, file)
        else:
            file.write(f"{key}: {value}\n")

def main(args):
    cfg.merge_from_file(args.config_file)


    # experiments_path = 'experiments/{}'.format(args.config_file.split('/')[1:-1])
    experiments_path = 'experiments/{}'.format('/'.join(args.config_file.split('/')[1:-1]))
    now_experiment_path = os.path.join(experiments_path,args.config_file.split('/')[-1].split('.')[0])
    now_ex_pseudo_masks_path = os.path.join(now_experiment_path,'pseudo_masks')
    now_ex_prototypes_path = os.path.join(now_experiment_path,'prototypes')
    now_ex_logs_path = os.path.join(now_experiment_path,'logs')
    now_ex_models_path = os.path.join(now_experiment_path,'models')
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)
    if not os.path.exists(now_experiment_path):
        os.makedirs(now_experiment_path)
    # if not os.path.exists(cfg.MODEL.save_path):
    #     os.makedirs(cfg.MODEL.save_path)
    if not os.path.exists(now_ex_models_path):
        os.makedirs(now_ex_models_path)
    if not os.path.exists(now_ex_pseudo_masks_path):
        os.makedirs(now_ex_pseudo_masks_path)
    if not os.path.exists(now_ex_prototypes_path):
        os.makedirs(now_ex_prototypes_path)
    if not os.path.exists(now_ex_logs_path):
        os.makedirs(now_ex_logs_path)


    cfg.MODEL.logs_path = now_ex_logs_path
    cfg.MODEL.save_path = now_ex_models_path
    cfg.freeze()
    # 保存当前配置文件
    cfg_path = os.path.join(now_experiment_path,'config.yaml')
    with open(cfg_path,'w') as f:
        yaml.dump(cfg,f)

    # tensorboard
    writer = SummaryWriter(os.path.join(now_ex_logs_path,'tf'))

    valset = SemiDataset(cfg.MODEL.task,cfg.MODEL.dataset, cfg.MODEL.data_root, 'val', None,cfg=cfg)
    valloader = DataLoader(valset, batch_size=4,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    criterion = CrossEntropyLoss(ignore_index=255)
    model, optimizer = init_basic_elems(cfg)

    MODE = 'train'
    trainset = SemiDataset(cfg.MODEL.task, cfg.MODEL.dataset, cfg.MODEL.data_root, MODE, cfg.MODEL.crop_size,
                           cfg.MODEL.labeled_id_path,cfg=cfg)
    trainloader = DataLoader(trainset, batch_size=cfg.MODEL.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)
    if cfg.MODEL.sup:
        print("==================>正在进行全监督训练,训练集全标记<=====================")
        train(MODE, model, trainloader, valloader, criterion, optimizer, cfg, writer)
    else:
        if cfg.MODEL.stage1:
            print("==================>正在进行全监督训练，在(标记数据部分)<=====================")
            MODE = 'train'
            best_model, checkpoints = train(MODE, model, trainloader, valloader, criterion, optimizer, cfg, writer)
        if cfg.MODEL.stage2:
            print("==================>正在计算源域数据的prototype<=====================")
            if cfg.MODEL.stage2_prototype:
                prototype_dist_init(cfg,src_train_loader=trainloader)
            print("==================>计算prototype结束<=====================")
            if cfg.MODEL.sup_uda:
                print("==================>正在进行无监督域适应训练<=====================")
                src_dataloader = trainloader
                tgt_trainset = SemiDataset(cfg.MODEL.task, cfg.MODEL.dataset, cfg.MODEL.data_root, MODE,
                                       cfg.MODEL.crop_size,
                                       cfg.MODEL.labeled_id_path_2,cfg=cfg)
                tgt_trainloader = DataLoader(tgt_trainset, batch_size=cfg.MODEL.batch_size, shuffle=True,
                                         pin_memory=True, num_workers=8, drop_last=True)
                model = load_model_ckpt(model,cfg.MODEL.stage1_ckpt_path)
                # print(model)
                src_tgt_train(MODE,model,src_dataloader,tgt_trainloader,valloader, optimizer, cfg, writer)
        if cfg.MODEL.stage3:
            print("==================>正在计算利用教师网络推理未标记数据伪标签<=====================")
            MODE = 'label'
            best_model = load_model_ckpt(model,cfg.MODEL.stage1_ckpt_path)
            tgt_trainset = SemiDataset(cfg.MODEL.task, cfg.MODEL.dataset, cfg.MODEL.data_root,MODE,
                                       None, None, cfg.MODEL.unlabeled_id_path,cfg=cfg)
            tgt_trainloader = DataLoader(tgt_trainset, batch_size=1, shuffle=False,
                                         pin_memory=True, num_workers=8, drop_last=False)
            label(best_model, tgt_trainloader, cfg)
            print("==================>伪标签推理结束！<=====================")
        if cfg.MODEL.stage4:
            MODE = 'semi_train'
            print("==================>训练学生网络<=====================")
            print("教师网络配置：{}".format(cfg.MODEL.stage1_ckpt_path))
            trainset = SemiDataset(cfg.MODEL.task,cfg.MODEL.dataset, cfg.MODEL.data_root, MODE, cfg.MODEL.crop_size,
                                   cfg.MODEL.labeled_id_path, cfg.MODEL.unlabeled_id_path, cfg.MODEL.pseudo_mask_path,cfg)
            trainloader = DataLoader(trainset, batch_size=cfg.MODEL.batch_size, shuffle=True,
                                     pin_memory=True, num_workers=16, drop_last=True)
            best_model = train(MODE, model, trainloader, valloader, criterion, optimizer, cfg, writer)

def init_basic_elems(cfg):

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[cfg.MODEL.model](cfg.MODEL.backbone,cfg.MODEL.NUM_CLASSES)

    head_lr_multiple = 10.0
    if cfg.MODEL.model == 'deeplabv2':
        assert cfg.MODEL.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg.MODEL.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': cfg.MODEL.lr * head_lr_multiple}],
                    lr=cfg.MODEL.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer






if __name__ == '__main__':
    args = parse_args()

    print(args)

    main(args)