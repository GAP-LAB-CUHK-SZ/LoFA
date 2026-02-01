import math
from typing import List, Union

import torch
import torch.nn.functional as F

def get_loss_fn(loss_type):
    if loss_type == "weighted_l1_mask_loss":
        return weighted_l1_loss_mask
    elif loss_type == "kl_loss":
        return kl_loss
    elif loss_type == "l1_loss":
        return l1_loss
    elif loss_type == "deltaW_l1_loss":
        return deltaW_l1_loss
    elif loss_type == "ce_loss":
        return ce_loss
    elif loss_type == "l2_penalty":
        return l2_penalty
    else:
        raise NotImplementedError

def ce_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_mask, gt_mask)

def l2_penalty(A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
    return (A.pow(2).mean() + B.pow(2).mean())

def l1_loss(pred_param: List[torch.Tensor], target_param: List[torch.Tensor], **kwargs) -> torch.Tensor:
    total_loss = 0.0
    for pred, target in zip(pred_param, target_param):
        total_loss += F.l1_loss(pred, target)
    return total_loss

def deltaW_l1_loss(pred_param: List[torch.Tensor], target_param: Union[List[torch.Tensor], torch.Tensor], **kwargs) -> torch.Tensor:
    alpha = kwargs.get("alpha", None)
    pred_deltaW = torch.bmm(pred_param[1], pred_param[0])
    if alpha is not None:
        pred_deltaW = pred_deltaW * alpha[:, None, None] 
    target_deltaW = torch.bmm(target_param[1], target_param[0]) if isinstance(target_param, list) else target_param

    if kwargs.get("weighted", False):
        w = 1.0 / (target_deltaW.abs() + 1e-6)
        total_loss = (w * (pred_deltaW - target_deltaW).abs()).mean()
    total_loss = F.l1_loss(pred_deltaW, target_deltaW)
    return total_loss
    
def weighted_l1_loss_mask(pred_param: List[torch.Tensor], target_param: List[torch.Tensor], weight=1000, thresh=0.1, **kwargs) -> torch.Tensor:
    pass

def kl_loss(pred_param: List[torch.Tensor], target_param: List[torch.Tensor], weight=1000, **kwargs):
    total_loss = 0.0
    for pred, target in zip(pred_param, target_param):
        p = F.softmax(pred * weight, dim=1)
        q = F.softmax(target * weight, dim=1)
        kl = torch.sum(p * (p.log() - q.log()), dim=1).mean()
        total_loss += kl
    return total_loss