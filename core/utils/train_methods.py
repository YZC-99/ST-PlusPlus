from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import os
from torchmetrics import JaccardIndex,Dice

from copy import deepcopy
from utils import count_params, meanIOU, color_map
from core.configs import cfg
from core.utils.loss import PrototypeContrastiveLoss
from core.utils.prototype_dist_estimator import prototype_dist_estimator
from core.utils.lovasz_loss import lovasz_softmax
from PIL import Image
from skimage import measure,draw
import numpy as np

def src_tgt_train(MODE,model,src_trainloader,tar_trainloader, valloader, optimizer, cfg,writer: SummaryWriter):
    # 损失函数：交叉熵损失、基于原型得对比损失
    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    pcl_criterion = PrototypeContrastiveLoss(cfg)
    print(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024
    feat_estimator = prototype_dist_estimator(feature_num=feature_num, cfg=cfg)
    if cfg.SOLVER.MULTI_LEVEL:
        out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    iters = 0
    total_iters = len(src_trainloader) * cfg.MODEL.epochs
    previous_best = 0.0


    # logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    if MODE == 'src_tgt_train':
        checkpoints = []

    for epoch in range(cfg.MODEL.epochs):
        print("\n==> Epoch %i, learning rate = %.6f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_loss_feat = 0.0
        total_loss_out = 0.0
        print("length of src_dataloader:{}".format(len(src_trainloader)))
        print("length of tar_trainloader:{}".format(len(tar_trainloader)))
        tbar = tqdm(list(zip(src_trainloader, tar_trainloader)))
        for i, ((src_input, src_label), (tgt_input,_)) in enumerate(tbar):
            optimizer.zero_grad()
            # 源域的数据与标签
            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()
            # 目标域的输入
            tgt_input = tgt_input.cuda(non_blocking=True)

            # 源域图片的大小
            src_size = src_input.shape[-2:]
            # 获取源域的高维特征和输出
            # batchsize*2048*64*64  ； batchsize*num_class*512*512
            src_feat = model.module.backbone.base_forward(src_input)[-1]
            src_out = model(src_input)
            # 获得目标域的高维特征和输出
            tgt_feat = model.module.backbone.base_forward(tgt_input)[-1]
            tgt_out = model(tgt_input)

            # supervision loss
            src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)
            if cfg.SOLVER.LAMBDA_LOV.lambda_lov > 0:
                pred_softmax = F.softmax(src_pred, dim=1)
                loss_lov = lovasz_softmax(pred_softmax, src_label, ignore=255)
                loss_sup = ce_criterion(src_pred, src_label) + cfg.SOLVER.LAMBDA_LOV.lambda_lov * loss_lov
            else:
                loss_sup = ce_criterion(src_pred, src_label)

            # source mask: downsample the ground-truth label
            # 获取源域高维特征的形状
            B, A, Hs_feat, Ws_feat = src_feat.size()
            src_feat_mask = F.interpolate(src_label.unsqueeze(0).float(), size=(Hs_feat, Ws_feat), mode='nearest').squeeze(0).long()
            src_feat_mask = src_feat_mask.contiguous().view(B * Hs_feat * Ws_feat, )
            assert not src_feat_mask.requires_grad
            # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
            _, _, Ht_feat, Wt_feat = tgt_feat.size()
            tgt_out_maxvalue, tgt_mask = torch.max(tgt_out, dim=1)
            for j in range(cfg.MODEL.NUM_CLASSES):
                tgt_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask == j)] = 255

            tgt_feat_mask = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_feat, Wt_feat), mode='nearest').squeeze(0).long()
            tgt_feat_mask = tgt_feat_mask.contiguous().view(B * Ht_feat * Wt_feat, )
            assert not tgt_feat_mask.requires_grad

            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs_feat * Ws_feat, A)
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht_feat * Wt_feat, A)
            # update feature-level statistics
            feat_estimator.update(features=tgt_feat.detach(), labels=tgt_feat_mask)
            feat_estimator.update(features=src_feat.detach(), labels=src_feat_mask)

            # contrastive loss on both domains
            loss_feat = pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                      feat=src_feat,
                                      labels=src_feat_mask) \
                        + pcl_criterion(Proto=feat_estimator.Proto.detach(),
                                        feat=tgt_feat,
                                        labels=tgt_feat_mask)

            if cfg.SOLVER.MULTI_LEVEL:
                _,_,Hs_out,Ws_out = src_out.size()
                _,_,Ht_out,Wt_out = tgt_out.size()
                src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, cfg.MODEL.NUM_CLASSES)
                tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht_out * Wt_out, cfg.MODEL.NUM_CLASSES)

                src_out_mask = src_label.unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(B * Hs_out * Ws_out, )
                tgt_pseudo_label = F.interpolate(tgt_mask.unsqueeze(0).float(), size=(Ht_out, Wt_out), mode='nearest').squeeze(0).long()
                tgt_out_mask = tgt_pseudo_label.contiguous().view(B * Ht_out * Wt_out, )


                # update output-level statistics
                out_estimator.update(features=tgt_out.detach(), labels=src_out_mask)
                out_estimator.update(features=src_out.detach(), labels=tgt_out_mask)

                # the proposed contrastive loss on prediction map
                loss_out = pcl_criterion(Proto=out_estimator.Proto.detach(),
                                         feat=src_out,
                                         labels=src_out_mask) \
                           + pcl_criterion(Proto=out_estimator.Proto.detach(),
                                           feat=tgt_out,
                                           labels=tgt_out_mask)

                loss = loss_sup \
                       + cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                       + cfg.SOLVER.LAMBDA_OUT * loss_out
                writer.add_scalar("src_tgt_train/step_loss_out", loss_out, iters + 1)
            else:
                loss = loss_sup + cfg.SOLVER.LAMBDA_FEAT * loss_feat

            loss.backward()
            writer.add_scalar("src_tgt_train/step_loss",loss,iters+1)
            writer.add_scalar("src_tgt_train/step_sup_loss",loss_sup,iters+1)
            writer.add_scalar("src_tgt_train/step_loss_feat",loss_feat,iters+1)

            optimizer.step()

            # 累计当前每个minibatch的损失
            total_loss += loss.item()
            total_sup_loss += loss_sup.item()
            total_loss_feat += loss_feat.item()
            if cfg.SOLVER.MULTI_LEVEL:
                total_loss_out += loss_out.item()
            iters += 1
            lr = cfg.MODEL.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if cfg.MODEL.model == 'deeplabv2' else lr * 10.0
            # 计算均值损失函数
            tbar.set_description('Loss: %.6f' % (total_loss / (i + 1)))

        writer.add_scalar("src_tgt_train/epoch_loss",total_loss / (i + 1),epoch)
        writer.add_scalar("src_tgt_train/epoch_sup_loss", total_sup_loss / (i + 1),epoch)
        writer.add_scalar("src_tgt_train/epoch_loss_feat", total_loss_feat / (i + 1),epoch)
        if cfg.SOLVER.MULTI_LEVEL:
          writer.add_scalar("src_tgt_train/epoch_loss_out", total_loss_out / (i + 1),epoch)
        metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)
        dice = Dice(num_classes=cfg.MODEL.NUM_CLASSES, average='macro')

        model.eval()
        tbar = tqdm(valloader)
        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                dice_score = dice(pred.cpu(), mask.cpu())
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            model_name = "uad_"
            mIOU *= 100.0
            dice_score *= 100.0
            writer.add_scalar("src_tgt_val/mIOU", mIOU, epoch)
            writer.add_scalar("src_tgt_val/Dice", dice_score, epoch)
            if mIOU > previous_best_IoU:
                if previous_best_IoU != 0:
                    os.remove(os.path.join(cfg.MODEL.save_path, '{}_mIoU{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                    cfg.MODEL.model, cfg.MODEL.backbone, previous_best_IoU)))
                previous_best_IoU = mIOU
                torch.save(model.module.state_dict(),
                           os.path.join(cfg.MODEL.save_path, '{}_mIoU{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                           cfg.MODEL.model, cfg.MODEL.backbone, mIOU)))

                best_model = deepcopy(model)

            if dice_score > previous_best_Dice:
                if previous_best_Dice != 0:
                    os.remove(os.path.join(cfg.MODEL.save_path, '{}_Dice{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                        cfg.MODEL.model, cfg.MODEL.backbone, previous_best_Dice)))
                previous_best_Dice = dice_score
                torch.save(model.module.state_dict(),
                           os.path.join(cfg.MODEL.save_path, '{}_Dice{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                               cfg.MODEL.model, cfg.MODEL.backbone, dice_score)))

            if MODE == 'train' and (
                    (epoch + 1) in [cfg.MODEL.epochs // 3, cfg.MODEL.epochs * 2 // 3, cfg.MODEL.epochs]):
                checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints
    return best_model

def draw_line(img_arr: np.array,gt: np.array,num_calssess):
    img_arr = img_arr.transpose((1,2,0))

    for cls in range(1, num_calssess+1):
        contours = measure.find_contours(gt==cls, 0.2)
        for contour in contours:
            for point in contour:
                x, y = point
                if cls == 1:
                    img_arr[int(x), int(y)] = [255, 0, 0]
                elif cls == 2:
                    img_arr[int(x), int(y)] = [0, 255, 0]
    return img_arr.transpose((2,0,1))

def train(MODE,model, trainloader, valloader, criterion, optimizer, cfg,writer: SummaryWriter):
    iters = 0
    total_iters = len(trainloader) * cfg.MODEL.epochs

    previous_best_IoU = 0.0
    previous_best_Dice = 0.0


    if MODE == 'train':
        checkpoints = []

    train_metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)
    for epoch in range(cfg.MODEL.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best_IoU))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("{}_train/step_loss_out".format(MODE), loss, iters + 1)
            total_loss += loss.item()

            iters += 1
            lr = cfg.MODEL.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if cfg.MODEL.model == 'deeplabv2' else lr * 10.0

            # tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))
            pred = torch.argmax(pred, dim=1)
            if iters % 2 == 0:
                image_with_mask = draw_line(np.array(img[0].cpu().squeeze()),np.array(mask[0].cpu()),cfg.MODEL.NUM_CLASSES)
                writer.add_image('images_train/image', img[0].squeeze(),iters)
                writer.add_image('images_train/masks', mask[0].unsqueeze(dim=0),iters)
                writer.add_image('images_train/image_with_mask', image_with_mask,iters)
                writer.add_image('images_train/preds', pred[0].unsqueeze(dim=0),iters)

            train_metric.add_batch(pred.detach().cpu().numpy(), mask.cpu().numpy())
            train_mIOU = train_metric.evaluate()[-1]
            tbar.set_description('train_mIOU: %.2f' % (train_mIOU * 100.0))
        writer.add_scalar("{}_train/epoch_loss_out".format(MODE), total_loss, epoch)
        metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)

        dice = Dice(num_classes=cfg.MODEL.NUM_CLASSES, average='macro')

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for idx,(img, mask, _) in enumerate(tbar):
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                dice_score = dice(pred.cpu(), mask.cpu())
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                if idx % 200 == 0:
                    writer.add_image('images_val/image', img[0].squeeze(), iters)
                    writer.add_image('images_val/masks', mask.cuda()[0].unsqueeze(dim=0), iters)
                    writer.add_image('images_val/preds', pred[0].unsqueeze(dim=0), iters)

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))



        if MODE == 'train' :
            model_name = "T"
        elif MODE == 'semi_train':
            model_name = "S"
        elif MODE == 'tgt_train':
            model_name = "S"
        mIOU *= 100.0
        dice_score *= 100.0
        writer.add_scalar("{}_val/mIOU".format(MODE), mIOU, epoch)
        writer.add_scalar("{}_val/Dice".format(MODE), dice_score, epoch)
        if mIOU > previous_best_IoU:
            if previous_best_IoU != 0:
                os.remove(os.path.join(cfg.MODEL.save_path, '{}_mIoU{}%s_%s_%.2f.pth'.format(MODE,model_name) % (cfg.MODEL.model, cfg.MODEL.backbone, previous_best_IoU)))
            previous_best_IoU = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(cfg.MODEL.save_path, '{}_mIoU{}%s_%s_%.2f.pth'.format(MODE,model_name) % (cfg.MODEL.model, cfg.MODEL.backbone, mIOU)))

            best_model = deepcopy(model)

        if dice_score > previous_best_Dice:
            if previous_best_Dice != 0:
                os.remove(os.path.join(cfg.MODEL.save_path, '{}_Dice{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                cfg.MODEL.model, cfg.MODEL.backbone, previous_best_Dice)))
            previous_best_Dice = dice_score
            torch.save(model.module.state_dict(),
                       os.path.join(cfg.MODEL.save_path, '{}_Dice{}%s_%s_%.2f.pth'.format(MODE, model_name) % (
                       cfg.MODEL.model, cfg.MODEL.backbone, dice_score)))

        if MODE == 'train' and ((epoch + 1) in [cfg.MODEL.epochs // 3, cfg.MODEL.epochs * 2 // 3, cfg.MODEL.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def label(model, dataloader, cfg):
    model.eval()
    tbar = tqdm(dataloader)
    metric = meanIOU(num_classes=cfg.MODEL.NUM_CLASSES)
    dice = Dice(num_classes=cfg.MODEL.NUM_CLASSES, average='macro')
    cmap = color_map(cfg.MODEL.dataset)

    # 创建csv文件
    with open(os.path.join(cfg.MODEL.logs_path,'pseudo_label_metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'IoU','Dice'])  # 写入表头
        with torch.no_grad():
            for img, mask, id in tbar:
                img = img.cuda()
                pred = model(img, True)
                pred = torch.argmax(pred, dim=1).cpu()

                metric.add_batch(pred.numpy(), mask.numpy())
                dice_score = dice(pred.cpu(), mask.cpu())
                mIOU = metric.evaluate()[-1]

                pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
                pred.putpalette(cmap)

                pred.save('%s/%s' % (cfg.MODEL.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

                # 写入csv
                writer.writerow([id[0],mIOU,dice_score.item()])
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
