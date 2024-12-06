# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : cal_loss.py
# @Project     : multi-semantic
# @CreateTime  : 2022/9/8 下午3:01:36
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
import torch
from torch import nn, autograd
import torch.nn.functional as F
from global_directions.content.encoder4editing.criteria.lpips.lpips import LPIPS
from global_directions.content.encoder4editing.criteria import id_loss, moco_loss

def cal_loss( y, y_hat, latent):
    id_lambda = 0.1
    l2_lambda = 1.0
    lpips_lambda = 0.8
    # Initialize loss
    lpips_loss = LPIPS(net_type="alex").to("cuda:0").eval()
    # id_loss = id_loss.IDLoss().to("cuda:0").eval()
    mse_loss = nn.MSELoss().to("cuda:0").eval()
    # Define loss
    loss = 0.0
    # # delta regularization loss
    # total_delta_loss = 0
    # deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()
    #
    # first_w = latent[:, 0, :]
    # for i in range(1, self.net.encoder.progressive_stage.value + 1):
    #     curr_dim = deltas_latent_dims[i]
    #     delta = latent[:, curr_dim, :] - first_w
    #     delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
    #     loss_dict[f"delta{i}_loss"] = float(delta_loss)
    #     total_delta_loss += delta_loss
    # loss_dict['total_delta_loss'] = float(total_delta_loss)
    # loss += self.opts.delta_norm_lambda * total_delta_loss

    # ### Similarity loss
    # loss_id, sim_improvement, id_logs = id_loss(y_hat, y, x)
    # loss += loss_id * id_lambda

    ### L2 loss
    loss_l2 = F.mse_loss(y_hat, y)
    loss += loss_l2 * l2_lambda

    ### LPIPS loss
    loss_lpips = lpips_loss(y_hat, y)
    loss += loss_lpips * lpips_lambda

    return loss