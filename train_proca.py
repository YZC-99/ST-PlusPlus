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
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from prototype_dist_init import prototype_dist_init
from core.configs import cfg
from core.utils.lovasz_loss import lovasz_softmax
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.misc import mkdir
from core.utils.loss import PrototypeContrastiveLoss
from core.utils.prototype_dist_estimator import prototype_dist_estimator

MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='refuge_od')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    parser.add_argument('--ckpt-path', type=str, default="./experiments/models/od_val_refuge/1_7/100refuge/Tdeeplabv3plus_resnet50_93.85.pth")

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: ' 'Teacher预训练已完成，直接进入第二阶段' % 6)

    global MODE
    MODE = 'train'

    # trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    # trainset.ids = 7 * trainset.ids if len(trainset.ids) < 700 else trainset.ids
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
    #                          pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    sd = torch.load(args.ckpt_path,map_location='cpu')
    new_state_dict = {}
    for key, value in sd.items():
        if not key.startswith('module.'):  # 如果关键字没有"module."前缀，加上该前缀
            key = 'module.' + key
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    """
        ST framework without selective re-training
    """

    # <======================== Re-training on labeled and unlabeled images ========================>
    print('\n\n\n================> Total stage 2/6: 第一次伪标签已推理，进入第三阶段')

    # <============================= 根据训练好的模型计算prototype  =============================>
    print('\n\n\n================> Total stage 3/6: 根据训练好的模型计算prototype')

    # output_dir = cfg.OUTPUT_DIR
    # if output_dir:
    #     mkdir(output_dir)
    # logger = setup_logger("prototype_dist_init", output_dir, 0)
    #
    # cfg.OUTPUT_DIR = 'experiments_pca/refuge'
    # cfg.CKPT_PATH = 'experiments/models/od_val_refuge/1_7/100refuge/Tdeeplabv3plus_resnet50_93.85.pth'
    # prototype_dist_init(cfg,src_train_loader = trainloader,logger=logger)

    print('\n\n\n================> Total stage 4/6: 利用源域真实标签与目标域伪标签训练T')
    src_trainset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path)
    src_trainset.ids = 7 * src_trainset.ids if len(src_trainset.ids) < 700 else src_trainset.ids
    src_trainloader = DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)

    tgt_dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    tgt_dataloader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    best_model, checkpoints = train(model,src_trainloader,tgt_dataloader,valloader,optimizer,args)
    print('\n\n\n================> Total stage 5/6: 利用S1进行二次标签推理')
    label(best_model,tgt_dataloader,args)
    print('\n\n\n================> Total stage 6/6: 第二次推理的标签单独训练S2')




    #
    # MODE = 'semi_train'
    #
    # trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
    #                        args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
    #                          pin_memory=True, num_workers=16, drop_last=True)
    #
    # model, optimizer = init_basic_elems(args)
    #
    # train(model, trainloader, valloader, criterion, optimizer, args)
    #
    # return



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


def train(model, src_trainloader,tar_trainloader, valloader, optimizer, args):


    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    pcl_criterion = PrototypeContrastiveLoss(cfg)
    # load init source prototypes
    # logger.info(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    iters = 0
    total_iters = len(src_trainloader) * args.epochs

    end = time.time()
    meters = MetricLogger(delimiter="  ")

    # logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    model.train()
    previous_best = 0.0
    global MODE
    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        '''
        src_trainloader的长度总是要小于tar_trainloader，假设src_trainloader长度100,tar_trainloader为700
        现在需要完成一个任务，zip(src_trainloader,tar_trainloader前100)、zip(src_trainloader,tar_trainloader的100-200)
        从而在后续的任务中让模型全部使用到tar_trainloader
        '''
        tbar = tqdm(zip(src_trainloader, tar_trainloader))
        for i, ((src_input, src_label), (tgt_input,_,_)) in enumerate(tbar):
            data_time = time.time() - end
            optimizer.zero_grad()
            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()
            tgt_input = tgt_input.cuda(non_blocking=True)
            # print()

            src_size = src_input.shape[-2:]
            src_feat = model.module.backbone.base_forward(src_input)[-1]
            src_out = model(src_input)
            tgt_feat = model.module.backbone.base_forward(tgt_input)[-1]
            tgt_out = model(tgt_input)
            # supervision loss
            src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)
            if cfg.SOLVER.LAMBDA_LOV > 0:
                pred_softmax = F.softmax(src_pred, dim=1)
                loss_lov = lovasz_softmax(pred_softmax, src_label, ignore=255)
                loss_sup = ce_criterion(src_pred, src_label) + cfg.SOLVER.LAMBDA_LOV * loss_lov
                meters.update(loss_lov=loss_lov.item())
            else:
                loss_sup = ce_criterion(src_pred, src_label)
            meters.update(loss_sup=loss_sup.item())

            # source mask: downsample the ground-truth label
            B, A, Hs, Ws = src_feat.size()
            src_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )
            assert not src_mask.requires_grad
            # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
            _, _, Ht, Wt = tgt_feat.size()
            tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
            for j in range(cfg.MODEL.NUM_CLASSES):
                tgt_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask == j)] = 255
            # print(tgt_mask.shape)
            # print(tgt_feat.shape)

            tgt_feat_mask = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
            tgt_feat_mask = tgt_feat_mask.contiguous().view(B * Ht * Wt, )
            assert not tgt_feat_mask.requires_grad

            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
            # update feature-level statistics
            feat_estimator.update(features=tgt_feat.detach(), labels=tgt_feat_mask)
            feat_estimator.update(features=src_feat.detach(), labels=src_mask)

            # contrastive loss on both domains

            loss_feat = pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                      feat=src_feat,
                                      labels=src_mask) \
                        + pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                        feat=tgt_feat,
                                        labels=tgt_feat_mask)
            meters.update(loss_feat=loss_feat.item())

            if cfg.SOLVER.MULTI_LEVEL:
                _,_,Hs_F,Ws_F = src_out.size()
                _,_,Ht_F,Wt_F = tgt_out.size()
                src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs_F * Ws_F, cfg.MODEL.NUM_CLASSES)
                tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht_F * Wt_F, cfg.MODEL.NUM_CLASSES)

                src_out_mask = src_label.unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(B * Hs_F * Ws_F, )
                tgt_pseudo_label = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_F, Wt_F), mode='nearest').squeeze(0).long()
                tgt_out_mask = tgt_pseudo_label.contiguous().view(B * Ht_F * Wt_F, )


                # update output-level statistics
                out_estimator.update(features=tgt_out.detach(), labels=src_out_mask)
                out_estimator.update(features=src_out.detach(), labels=tgt_out_mask)

                # the proposed contrastive loss on prediction map
                # print(src_feat.shape)
                # print(src_mask.shape)
                loss_out = pcl_criterion(Proto=out_estimator.Proto.detach(),
                                         feat=src_out,
                                         labels=src_out_mask) \
                           + pcl_criterion(Proto=out_estimator.Proto.detach(),
                                           feat=tgt_out,
                                           labels=tgt_out_mask)
                meters.update(loss_out=loss_out.item())

                loss = loss_sup \
                       + cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                       + cfg.SOLVER.LAMBDA_OUT * loss_out
            else:
                loss = loss_sup + cfg.SOLVER.LAMBDA_FEAT * loss_feat

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

            # batch_time = time.time() - end
            # end = time.time()
            # meters.update(time=batch_time, data=data_time)

        metric = meanIOU(num_classes=args.num_classes)
        model.eval()
        tbar = tqdm(valloader)
        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            if MODE == 'train' :
                model_name = "T"
            elif MODE == 'semi_train':
                model_name = "S"
            mIOU *= 100.0
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '{}%s_%s_%.2f.pth'.format(model_name) % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, '{}%s_%s_%.2f.pth'.format(model_name) % (args.model, args.backbone, mIOU)))

                best_model = deepcopy(model)

            if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
                checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints
    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=args.num_classes)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)
    metric = meanIOU(num_classes=args.num_classes)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))



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