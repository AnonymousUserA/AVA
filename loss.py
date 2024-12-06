# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : loss.py
# @Project     : multi-semantic
# @CreateTime  : 2022/3/23 下午1:42:38
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
import torch
import numpy as np
from torch import nn
def nontarget_logit_loss(logit, label, nclasses):
    # Dummy vector for one-hot label vector. For multi-class, change this to # of classes
    Y_ = torch.zeros(1, nclasses)
    Y_[0, label] = 1.0
    actual_logits = (Y_*logit).sum(1)
    nonactual_logits = ((1-Y_)*logit - Y_*10000).max(1)[0]
    model_loss = torch.clamp(actual_logits - nonactual_logits, min=0.0).sum()
    return model_loss