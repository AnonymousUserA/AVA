# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : get_inversion.py
# @Project     : multi-semantic
# @CreateTime  : 2022/5/7 上午9:43:39
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : attribute_edit_pytorch.py
# @Project     : multi-semantic
# @CreateTime  : 2022/4/7 下午5:02:47
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
# -*- coding: utf-8 -*-
import argparse
import math
import os
import sys
import torch
import torchvision
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.utils as vutils
import cv2
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
from matplotlib import pyplot as plt

from global_directions.manipulate import Manipulator
from target_models.networks.resnet import resnet50
from torchvision import transforms
from GetCode_real import get_real_latent
from MapTS import GetFs, GetBoundary, GetDt
from target_models.networks.resnet import resnet50
from torchvision import transforms
from models.stylegan2.model import Generator
from all_attack import all_attack
from network.models import model_selection
import torch.nn.functional as F
from loss import nontarget_logit_loss
from target_model.Xception.trainer import Trainer


def get_inversion(g_ema, image_path, output_path):
    image_name = image_path.split("/")[-1]

    print("You are working in:", image_name)
    # Get Latent
    latent_code_init = get_real_latent(image_path)
    if latent_code_init is None:
        return
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
    img_gen, _ = g_ema([dlatents_loaded], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)

    vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, image_name), padding=0)

if __name__ == '__main__':

    # define Methods
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    M = Manipulator(dataset_name='ffhq')
    # load latent
    dir_path = "/8T/work/search/dataset/adversarial/new/100/semantic_real/test"
    output_path = "/8T/work/search/dataset/defense/dip/inversion_real/semantic"
    os.makedirs(output_path, exist_ok=True)
    for image in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image)
        get_inversion(g_ema, image_path, output_path)