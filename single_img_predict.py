import os
import cv2
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model.centernet import CenterNet
from utils import pool_nms, inference_preprocess
from config import Config

import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NUM_CLASSES', default=80)
    parser.add_argument('--MODEL_DEPTH', default=34)
    parser.add_argument('--LOAD_FROM', default='ckpt/ckpt.pth')
    parser.add_argument('--LABEL_PATH', default='dataset/coco_classes.txt')
    parser.add_argument('--IMAGE_ROOT', default='dataset/imgs/')

    parser.add_argument('--inference_nums', default=30)
    parser.add_argument('--score_thresh', default=0.35)
    parser.add_argument('--nms_thresh', default=0.4)

    return parser

def predict(args):
    # 加载标签
    with open(args.LABEL_PATH, 'r') as f:
        LABELS = f.readlines()
    LABELS = list(map(lambda x: x[:-1], LABELS))
    COLOR_MAP = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(len(LABELS))]

    model = CenterNet(depth=args.MODEL_DEPTH, num_classes=args.NUM_CLASSES)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.LOAD_FROM))
    model.eval()
    # grid
    yv, xv = torch.meshgrid(torch.arange(0, 128), torch.arange(0, 128))
    xv, yv = xv.flatten().float(), yv.flatten().float()
    if torch.cuda.is_available():
        xv = xv.cuda()
        yv = yv.cuda()

    img_lists = os.listdir(args.IMAGE_ROOT)
    random.shuffle(img_lists)
    img_lists = img_lists[:args.inference_nums]

    for per_img in img_lists:
        start = time.time()

        img_path = os.path.join(args.IMAGE_ROOT, per_img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img.shape
        inputs = inference_preprocess(img)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        heatmap, wh, offset = model(inputs)
        # single image
        heatmap = pool_nms(heatmap, kernel=3)
        heatmap = heatmap[0]
        wh = wh[0]
        offset = offset[0]

        heatmap = heatmap.permute(1, 2, 0).reshape(-1, args.NUM_CLASSES)
        wh = wh.permute(1, 2, 0).reshape(-1, 2)
        offset = offset.permute(1, 2, 0).reshape(-1, 2)

        class_confs, class_preds = torch.max(heatmap, dim=-1)
        mask = (class_confs >= args.score_thresh)
        wh_preds = wh[mask]
        offset_preds = offset[mask]
        classes_preds = class_preds[mask]
        scores_preds = class_confs[mask]

        wh_preds *= 128

        half_w, half_h = wh_preds[..., 0] / 2, wh_preds[..., 1] / 2
        X = xv[mask]
        Y = yv[mask]
        X = X + offset_preds[:, 0]
        Y = Y + offset_preds[:, 1]

        boxes = torch.stack([X - half_w, Y - half_h, X + half_w, Y + half_h], dim=1)
        boxes /= 128  # map pred to [0, 1]

        boxes[:, [0, 2]] *= img_w
        boxes[:, [1, 3]] *= img_h

        # 规整超出边界的矩形框
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, img_w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, img_h - 1)

        keep = nms(boxes, scores_preds, iou_threshold=args.nms_thresh)

        boxes = boxes[keep]
        classes = classes_preds[keep]
        scores = scores_preds[keep]

        if torch.cuda.is_available():
            boxes = boxes.cpu()
            classes = classes.cpu()
            scores = scores.cpu()
        
        boxes = boxes.detach().numpy()
        classes = classes.detach().numpy()
        scores = scores.detach().numpy()

        elapse = time.time() - start
        print('fps: {:.2f}'.format(1 / elapse))

        # # imshow
        plt.figure(dpi=120)
        plt.imshow(img)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            catId = classes[i]
            className = LABELS[catId]
            score = scores[i]
            plt.gca().add_patch(patches.Rectangle((x1, y1), w, h, color=COLOR_MAP[catId], fill=False))
            msg = "{}:{:.2f}".format(className, score)
            plt.text(x1, y1-5, msg,
                     bbox=dict(facecolor=COLOR_MAP[catId], edgecolor=COLOR_MAP[catId], boxstyle='round'),
                     fontdict={'size': 6})
        
        plt.axis('off')
        plt.savefig('log/inference' + '/' + per_img)
        plt.show()


if __name__ == '__main__':
    parse = parse_args()
    args = parse.parse_args()
    predict(args)