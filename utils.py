import cv2
import collections
import torch
import torch.nn as nn
import numpy as np

def inference_resize(img, resize_size=512, fill=False):
    """
    return resized img
    """
    img_h, img_w, _ = img.shape
    if fill:
        max_side = max(img_h, img_w)
        ratio = resize_size / max_side

        new_h, new_w = int(ratio * img_h), int(ratio * img_w)
        resized_img = cv2.resize(img, (new_w, new_h))

        new_img = np.zeros(shape=(resize_size, resize_size, 3),dtype=np.uint8)
        new_img[:new_h, :new_w, :] = resized_img
        return new_img
    else:
        new_img = cv2.resize(img, (resize_size, resize_size))
        return new_img
    

def pool_nms(hm, kernel=3):
    """
    on location (x, y), if hm[x, y] > others 8 points, then it can be seem as a keypoint
    """
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == hm).float()
    return hm * keep


def inference_preprocess(img, resize_size=512, fill=False):
    inputs = inference_resize(img, resize_size=resize_size, fill=fill)
    inputs = inputs / 255.
    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.permute(2, 0, 1)
    inputs = inputs.unsqueeze(dim=0)
    return inputs


def load_model(model, state_dict_path):
    """
    cause the final fc layer tensor shape may different when load pretrained model
    for same shape use same shape tensor
    """
    pretrain_state_dict = torch.load(state_dict_path)
    model_state_dict = model.state_dict()
    update_state_dict = collections.OrderedDict()

    for key in model_state_dict:
        if model_state_dict[key].shape == pretrain_state_dict[key].shape:
            update_state_dict[key] = pretrain_state_dict[key]
        else:
            update_state_dict[key] = model_state_dict[key]
    
    model.load_state_dict(update_state_dict)
    return model
