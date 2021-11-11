import os
from cv2 import resize
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimize
from torch.utils import data
from dataset.dataset import COCODataset, VOCDataset

from model.centernet import CenterNet
from model.loss import focal_loss, reg_l1_loss
from utils import load_model
from config import Config

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'


def main(cfg):
    # dataset
    if cfg.dataset_format == 'coco':
        TrainDataset = COCODataset(cfg.train_data_root, cfg.train_label_path, cfg.input_size)
        TrainDataloader = data.DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        TrainDataset = VOCDataset(cfg.voc_root, resize_size=cfg.input_size)
        TrainDataloader = data.DataLoader(TrainDataset, batch_size=cfg.batch_size, shuffle=True)

    # model
    model = CenterNet(depth=cfg.depth, num_classes=cfg.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)
    
    # pretained load
    model = load_model(model, cfg.load_from)
    
    # optim
    optimizer = optimize.Adam(model.parameters(), lr=cfg.lr)
    
    # lr schedule
    lr_scheduler = optimize.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    iters = len(TrainDataloader)
    for epoch in range(cfg.epoches):
        epoch_total_loss = []
        epoch_class_loss = []
        epoch_wh_loss = []
        epoch_offset_loss = []

        for step, sample in enumerate(TrainDataloader):
            img, hm, wh, offset, location_mask = sample
            if torch.cuda.is_available():
                img, hm, wh, offset, location_mask = img.cuda(), hm.cuda(), wh.cuda(), offset.cuda(), location_mask.cuda()

            pred_hm, pred_wh, pred_offset = model(img)

            class_loss = focal_loss(pred_hm, hm)
            wh_loss = 0.1 * reg_l1_loss(pred_wh, wh, location_mask)
            offset_loss = reg_l1_loss(pred_offset, offset, location_mask)
            total_loss = class_loss + wh_loss + offset_loss

            # epoch loss
            epoch_total_loss.append(total_loss.item())
            epoch_class_loss.append(class_loss.item())
            epoch_wh_loss.append(wh_loss.item())
            epoch_offset_loss.append(offset_loss.item())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            msg = "resnet34 coco, epoch: {}/{}, iter: {}/{}, class_loss: {:.5f}, wh_loss: {:.5f}, offset_loss: {:.5f}, total_loss: {:.5f}"\
                .format(epoch + 1, cfg.epoches, step+1, iters, np.mean(epoch_class_loss), np.mean(epoch_wh_loss), np.mean(epoch_offset_loss), np.mean(epoch_total_loss))
            print(msg)
        
        lr_scheduler.step(np.mean(epoch_total_loss))
        # save ckpt
        save_path = os.path.join(cfg.ckpt_path, "epoch_{}_loss_{:.2f}.pth".format(epoch + 1, np.mean(epoch_total_loss)))
        torch.save(model.state_dict(), save_path)
        print("model saved at: {}".format(save_path))


if __name__ == '__main__':
    cfg = Config()
    main(cfg)