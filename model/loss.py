import torch
import torch.nn.functional as F


def focal_loss(pred, target):
    """
    pred: [B, C, H, W]
    target: [B, H, W, C]
    """
    pred = pred.permute(0, 2, 3, 1)

    # 获取正样本的索引
    pos_inds = target.eq(1).float()
    # 负样本索引
    neg_inds = target.lt(1).float()
    
    # 负样本损失权重，离目标的中心点越远，是负样本的概率应该越大
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    
    # 正样本损失
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    
    pred = pred.permute(0, 2, 3, 1) 
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
    # 只有正样本位置才计算bounding box的回归损失
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss