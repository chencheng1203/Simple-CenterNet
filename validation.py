import os
import cv2
import random
import time
import json
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval


import torch
import torch.nn as nn
from torchvision.ops import nms, batched_nms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model.centernet import CenterNet
from dataset.dataset import COCODataset
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
    parser.add_argument('--JSON_LABEL_PATH', default='/home/workspace/chencheng/code/dataset/MSCOCO/annotations/instances_val2014.json')
    parser.add_argument('--IMAGE_ROOT', default='/home/workspace/chencheng/code/dataset/MSCOCO/val2014')

    parser.add_argument('--score_thresh', default=0.4)
    parser.add_argument('--nms_thresh', default=0.4)

    return parser


def generate_coco_results(args):

    model = CenterNet(depth=args.MODEL_DEPTH, num_classes=args.NUM_CLASSES)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.LOAD_FROM))
    model.eval()

    # load dataset
    with open(args.JSON_LABEL_PATH, 'r') as f:
        json_label = json.load(f)

    sequence2coco_catId = {}
    for i, perCat in enumerate(json_label['categories']):
        sequence2coco_catId[i] = perCat['id']

    results = []
    
    # grid
    yv, xv = torch.meshgrid(torch.arange(0, 128), torch.arange(0, 128))
    xv, yv = xv.flatten().float(), yv.flatten().float()
    if torch.cuda.is_available():
        xv = xv.cuda()
        yv = yv.cuda()

    for per_img in tqdm(json_label['images']):
        imgName = per_img['file_name']
        imgId = per_img['id']

        img_path = os.path.join(args.IMAGE_ROOT, imgName)
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
        boxes[:, [0, 2]] = torch.clip(boxes[:, [0, 2]], 0, img_w - 1)
        boxes[:, [1, 3]] = torch.clip(boxes[:, [1, 3]], 0, img_h - 1)

        keep = nms(boxes, scores_preds, iou_threshold=args.nms_thresh)
        # keep = batched_nms(boxes, scores_preds, classes_preds, iou_threshold=args.nms_thresh)

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

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            new_box = [int(x1), int(y1), int(w), int(h)]
            cat_id = sequence2coco_catId[classes[i]]
            image_result = {
                            'image_id'    : imgId,
                            'category_id' : cat_id,
                            'score'       : float(scores[i]),
                            'bbox'        : new_box,
                        }
            results.append(image_result)
    
    # write output
    log_name = args.LOAD_FROM.split('/')[-1][:8] + '.json'
    log_save_path = os.path.join('log/eval', log_name)
    json.dump(results, open(log_save_path, 'w'), indent=4)
    print('eval log saved at: {}'.format(log_save_path))
    return log_save_path

def eval(args, log_save_path):

    ValDataset = COCODataset(args.IMAGE_ROOT, args.JSON_LABEL_PATH)
    coco_true = ValDataset.coco
    coco_pred = coco_true.loadRes(log_save_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
if __name__ == '__main__':
    parse = parse_args()
    args = parse.parse_args()
    log_save_path = generate_coco_results(args)
    eval(args, log_save_path)