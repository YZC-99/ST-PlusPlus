from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')
    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')


    parser.add_argument('--ckpt-path', type=str, default='./experiments/models/Train_GAMMA_Val_REFUGE/1_8/Sdeeplabv3plus_resnet50_0.98.pth')

    args = parser.parse_args()
    return args


def main(args):

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)



    model, optimizer = init_basic_elems(args)

    sd = torch.load(args.ckpt_path,map_location='cpu')
    new_state_dict = {}
    for key, value in sd.items():
        if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
            key = 'module.' + key
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

    model.eval()
    model.cuda()
    # print(model)
    metric = meanIOU(args.num_classes)

    tbar = tqdm(valloader)

    with torch.no_grad():
        for img, mask, _ in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
    mIOU *= 100.0
    print(mIOU)





def init_basic_elems(args):

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone,args.num_classes)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer






if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = 240
    if args.lr is None:
        args.lr = 0.004 / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = 512

    print()
    print(args)

    main(args)