import os
import cv2
import json
import torch
import albumentations as A
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pycocotools.coco import COCO
from torch.utils import data
import xml.etree.ElementTree as ET


class COCODataset(data.Dataset):
    def __init__(self, data_root, label_path, resize_size=512):
        super(COCODataset, self).__init__()
        self.data_root = data_root
        self.resize_size = resize_size
        self.output_size = self.resize_size // 4

        with open(label_path, 'r') as f:
            json_label = json.load(f)
        # 使用pycocotools对标签进行解析
        self.coco = COCO(label_path)
        self.img_ids = self.coco.getImgIds()

        # 由于coco的标签id并不是连续的，所以需要映射一下
        self.cocoId2SequenceId = {}
        for i, per_cat in enumerate(json_label['categories']):
            self.cocoId2SequenceId[per_cat['id']] = i

        self.num_classes = len(self.cocoId2SequenceId)

        # 数据增强
        self.transform = A.Compose([A.HorizontalFlip(p=0.5),
                                    A.RandomBrightnessContrast(p=0.2)],
                                    bbox_params=A.BboxParams(format='coco'))
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        imgId = self.img_ids[index]
        imgInfo = self.coco.loadImgs(imgId)[0]
        imgName = imgInfo['file_name']
        # imgHeight, imgWidth = imgInfo['height'], imgInfo['width']
        img = cv2.imread(os.path.join(self.data_root, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annots = self.load_annots(imgId)

        if len(annots) != 0:
            # 数据增强
            transformed = self.transform(image=img, bboxes=annots)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'])

            # 固定尺寸的输入，需要resize
            img, boxes = self.resize(img, boxes)
            
            hm, wh, offset, location_mask = self.generate_targets(boxes)
            hm = torch.from_numpy(hm)
            wh = torch.from_numpy(wh)
            offset = torch.from_numpy(offset)
            location_mask = torch.from_numpy(location_mask)
        else:
            img = cv2.resize(img, (self.resize_size, self.resize_size))
            hm = torch.zeros(size=(self.output_size, self.output_size, self.num_classes))
            wh = torch.zeros(size=(self.output_size, self.output_size, 2))
            offset = torch.zeros(size=(self.output_size, self.output_size, 2))
            location_mask = torch.zeros(size=(self.output_size, self.output_size), dtype=bool)
        img = self.to_tentor(img)
        return img, hm, wh, offset, location_mask
    
    def load_annots(self, imgId):
        annIds = self.coco.getAnnIds(imgId) 
        anns = self.coco.loadAnns(annIds)

        boxes = []
        for per_box in anns:
            catId = self.cocoId2SequenceId[per_box['category_id']]
            x1, y1, w, h = per_box['bbox']
            if (w == 0 or h == 0):
                continue
            boxes.append([x1, y1, w, h, catId])
        return np.array(boxes)
    
    def resize(self, img, boxes, resize_size=512, fill=False):
        '''
        fill: 是否填充黑边，保持原图长宽比的一致性
        '''
        img_h, img_w, _ = img.shape
        if fill:
            # 将最大的边长resize到目标尺寸
            max_side = max(img_h, img_w)
            ratio = resize_size / max_side

            new_h, new_w = int(ratio * img_h), int(ratio * img_w)
            resized_img = cv2.resize(img, (new_w, new_h))

            new_img = np.zeros(shape=(resize_size, resize_size, 3), dtype=np.uint8)
            new_img[:new_h, :new_w, :] = resized_img

            boxes[:, :4] = boxes[:, :4] * ratio
            return new_img, boxes
        else:
            x_ratio = resize_size / img_w
            y_ratio = resize_size / img_h
            new_img = cv2.resize(img, (resize_size, resize_size))
            # [x, y, w, h]
            boxes[:, 0] = boxes[:, 0] * x_ratio
            boxes[:, 2] = boxes[:, 2] * x_ratio
            boxes[:, 1] = boxes[:, 1] * y_ratio
            boxes[:, 3] = boxes[:, 3] * y_ratio
            return new_img, boxes
        
    def generate_targets(self, boxes):
        '''
        生成标签
        '''
        # heatmap: [W, H, num_classes]
        hm = np.zeros(shape=(self.output_size, self.output_size, self.num_classes))
        # wh: [W, H, 2]
        wh = np.zeros(shape=(self.output_size, self.output_size, 2))
        # offset: [W, H, 2]
        offset = np.zeros(shape=(self.output_size, self.output_size, 2))
        # mask: [W, H] 表示该位置的feature map上是否出现了目标
        location_mask = np.zeros(shape=(self.output_size, self.output_size), dtype=bool)
        # 将ground-truth 缩放到下采样的尺度
        boxes[:, :4] = boxes[:, :4] // 4
        for per_box in boxes:
            x1, y1, w, h, cat_id = per_box
            # 将中心点坐标转换成整型
            ctr_x = int(x1 + w / 2)
            ctr_y = int(y1 + h / 2)
            cat_id = int(cat_id)
            # mask
            # 该位置存在目标
            location_mask[ctr_x, ctr_y] = 1
            # hm
            # 以该点为中心绘制二维高斯核
            hm[:, :, cat_id] = self.draw_guassian(hm[:, :, cat_id], ctr_x, ctr_y, w, h)
            # wh
            # 计算该中心点的目标的长和宽，归一化到0-1之间
            wh[ctr_x, ctr_y, 0] = w / self.output_size
            wh[ctr_x, ctr_y, 1] = h / self.output_size
            # offset
            # 宽度和长度方向上的偏移量
            offset[ctr_x, ctr_y, 0] = x1 + w / 2 - ctr_x
            offset[ctr_x, ctr_y, 1] = y1 + h / 2 - ctr_y

        return hm, wh, offset, location_mask
    
    def draw_guassian(self, hm, ctr_x, ctr_y, w, h):
        """
        hm: single category heat map with shape [w, h]
        """
        X = np.linspace(0, self.output_size, self.output_size, dtype=int)
        Y = np.linspace(0, self.output_size, self.output_size, dtype=int)
        x, y = np.meshgrid(X, Y)

        # 计算高斯核的半径
        radius = self.gaussian_radius(h, w)
        diameter = 2 * radius + 1
        sigma = diameter / 5
        curr_guassian = np.exp(-((x - ctr_x) ** 2 + (y - ctr_y) ** 2) / (2 * sigma ** 2))
        # 同一个位置取最大值
        hm = np.maximum(hm, curr_guassian)
        return hm

    def gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2
        
        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)
    
    def to_tentor(self, x):
        if np.max(x) > 1:
            x = x / 255.
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)
        return x.float()

    def get_num_classes(self):
        return self.num_classes


class VOCDataset(data.Dataset):
    def __init__(self, voc_root, resize_size=512):
        super(VOCDataset, self).__init__()
        self.num_classes = 20
        self.resize_size = resize_size
        self.output_size = self.resize_size // 4

        self.img_root = os.path.join(voc_root, 'JPEGImages')
        self.label_root = os.path.join(voc_root, 'Annotations')
        selected_img_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'train.txt')

        with open(selected_img_txt_path, 'r') as f:
            self.img_lists = f.readlines()
        self.img_lists = list(map(lambda x: x[:-1], self.img_lists))
    
        self.catName2id = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
                            'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
                            'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,'sheep': 16, 
                            'sofa': 17, 'train': 18, 'tvmonitor': 19}

        # 数据增强
        self.transform = A.Compose([A.HorizontalFlip(p=0.5),
                                    A.RandomBrightnessContrast(p=0.2)],
                                    bbox_params=A.BboxParams(format='coco'))

    def __len__(self):
        return len(self.img_lists)
    
    def __getitem__(self, index):
        img_name = self.img_lists[index]
        img = cv2.imread(os.path.join(self.img_root, img_name + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annots = self.load_annots(img_name)
        
        if len(annots) != 0:
            img, boxes = self.resize(img, annots, resize_size=self.resize_size)

            # 数据增强
            # transformed = self.transform(image=img, bboxes=boxes)
            # img = transformed['image']
            # boxes = np.array(transformed['bboxes'])

            hm, wh, offset, location_mask = self.generate_targets(boxes)
            hm = torch.from_numpy(hm)
            wh = torch.from_numpy(wh)
            offset = torch.from_numpy(offset)
            location_mask = torch.from_numpy(location_mask)
        else:
            img = cv2.resize(img, (self.resize_size, self.resize_size))
            hm = torch.zeros(size=(self.output_size, self.output_size, self.num_classes))
            wh = torch.zeros(size=(self.output_size, self.output_size, 2))
            offset = torch.zeros(size=(self.output_size, self.output_size, 2))
            location_mask = torch.zeros(size=(self.output_size, self.output_size), dtype=bool)
        img = self.to_tentor(img)
        return img, hm, wh, offset, location_mask

    def load_annots(self, img_name):
        xml_path = os.path.join(self.label_root, img_name + '.xml')
        et = ET.parse(xml_path)
        element = et.getroot()
        element_objs = element.findall('object')
        boxes = []
        for per_obj in element_objs:
            cat_name = per_obj.find('name').text
            obj_bbox = per_obj.find('bndbox')
            x1 = float(obj_bbox.find('xmin').text)
            y1 = float(obj_bbox.find('ymin').text)
            x2 = float(obj_bbox.find('xmax').text)
            y2 = float(obj_bbox.find('ymax').text)
            w = x2 - x1
            h = y2 - y1
            catId = self.catName2id[cat_name]
            boxes.append([x1, y1, w, h, catId])
        return np.array(boxes)

    def resize(self, img, boxes, resize_size=512, fill=False):
        img_h, img_w, _ = img.shape
        if fill:
            max_side = max(img_h, img_w)
            ratio = resize_size / max_side

            new_h, new_w = int(ratio * img_h), int(ratio * img_w)
            resized_img = cv2.resize(img, (new_w, new_h))

            new_img = np.zeros(shape=(resize_size, resize_size, 3),dtype=np.uint8)
            new_img[:new_h, :new_w, :] = resized_img

            boxes[:, :4] = boxes[:, :4] * ratio
            return new_img, boxes
        else:
            x_ratio = resize_size / img_w
            y_ratio = resize_size / img_h
            new_img = cv2.resize(img, (resize_size, resize_size))
            # [x, y, w, h]
            boxes[:, 0] = boxes[:, 0] * x_ratio
            boxes[:, 2] = boxes[:, 2] * x_ratio
            boxes[:, 1] = boxes[:, 1] * y_ratio
            boxes[:, 3] = boxes[:, 3] * y_ratio
            return new_img, boxes

    def generate_targets(self, boxes):
        hm = np.zeros(shape=(self.output_size, self.output_size, self.num_classes))
        wh = np.zeros(shape=(self.output_size, self.output_size, 2))
        offset = np.zeros(shape=(self.output_size, self.output_size, 2))
        location_mask = np.zeros(shape=(self.output_size, self.output_size), dtype=bool)
        
        boxes[:, :4] = boxes[:, :4] // 4
        for per_box in boxes:
            x1, y1, w, h, cat_id = per_box
            ctr_x = int(x1 + w / 2)
            ctr_y = int(y1 + h / 2)
            cat_id = int(cat_id)
            # mask
            location_mask[ctr_x, ctr_y] = 1
            # hm
            hm[:, :, cat_id] = self.draw_guassian(hm[:, :, cat_id], ctr_x, ctr_y, w, h)
            # wh
            wh[ctr_x, ctr_y, 0] = w / self.output_size
            wh[ctr_x, ctr_y, 1] = h / self.output_size
            # offset
            offset[ctr_x, ctr_y, 0] = x1 + w / 2 - ctr_x
            offset[ctr_x, ctr_y, 1] = y1 + h / 2 - ctr_y

        return hm, wh, offset, location_mask
    
    def draw_guassian(self, hm, ctr_x, ctr_y, w, h):
        """
        hm: single category heat map with shape [w, h]
        """
        X = np.linspace(0, self.output_size, self.output_size, dtype=int)
        Y = np.linspace(0, self.output_size, self.output_size, dtype=int)
        x, y = np.meshgrid(X, Y)

        radius = self.gaussian_radius(h, w)
        diameter = 2 * radius + 1
        sigma = diameter / 6
        curr_guassian = np.exp(-((x - ctr_x) ** 2 + (y - ctr_y) ** 2) / (2 * sigma ** 2))
        hm = np.maximum(hm, curr_guassian)
        return hm

    def gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2
        
        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)
    
    def to_tentor(self, x):
        if np.max(x) > 1:
            x = x / 255.
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)
        return x.float()

if __name__ == '__main__':
    data_root = '/home/workspace/chencheng/code/dataset/MSCOCO/train2014'
    label_path = '/home/workspace/chencheng/code/dataset/MSCOCO/annotations/instances_train2014.json'
    voc_root = '/home/workspace/chencheng/code/dataset/VOC2007'
    dataset = COCODataset(data_root, label_path)
    for i in tqdm(range(len(dataset))):
        dataset[i]