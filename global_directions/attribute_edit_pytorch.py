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
# from target_models.networks.resnet import resnet50
from torchvision import transforms
from GetCode_real import get_real_latent
from MapTS import GetFs, GetBoundary, GetDt
# from target_models.networks.resnet import resnet50
from torchvision import transforms
from models.stylegan2.model import Generator
from all_attack import all_attack
from network.models import model_selection
import torch.nn.functional as F
# from loss import nontarget_logit_loss
from target_model.Xception.trainer import Trainer
import cal_loss

def denorm1(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def run_alignment(image_path):
    import dlib
    from global_directions.content.encoder4editing.utils.alignment import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def get_clostest_latent(g_ema, M, image_path, output_path):
    image_name = image_path.split("/")[-1]
    latent_code_init = get_real_latent(image_path)
    if latent_code_init is None:
        return
    output, _ = g_ema([latent_code_init], input_is_latent=True)
    vutils.save_image(((output + 1) / 2).data, os.path.join(output_path, image_name), padding=0)

def oneEdit_oneAttr_oneStyle(g_ema, M, image_name, latent_code_init, output_path, attr, lindex, cindex):
    """ One edit strength for specific attribute though (lindex, cindex) and save to output_path
    Args:
        g_ema:
        M:
        image_path:
        output_path:
        attr:
        lindex:
        cindex:

    Returns:

    """
    # file and dir preprocess
    print("You are working in:", image_name)
    os.makedirs(output_path, exist_ok=True)


    # Get S
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
    latent = [s.detach().clone() for s in dlatents_loaded]

    # edit specific (lindex, cindex) of stylespace
    alpha = 0
    latent[lindex][0][0][cindex][0][0] = latent[lindex][0][0][cindex][0][0] + alpha

    # generate edited image
    img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)

    # save tensor of edited image to png
    vutils.save_image(((img_gen + 1) / 2).clamp_(0, 1).data, os.path.join(output_path, image_name), padding=0)

def oneEdit_manyAttr_manyStyle(g_ema, M, image_name, latent_code_init, output_path, config):
    # file and dir preprocess
    print("You are working in:", image_name)
    os.makedirs(output_path, exist_ok=True)

    # set edit stregth
    alpha = 20  # hair

    # Get S
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)

    # edit specific (lindex, cindex) of stylespace
    for value in config:
        latent = [s.detach().clone() for s in dlatents_loaded]
        lindex, cindex = value[0], value[1]
        latent[lindex][0][0][cindex][0][0] = latent[lindex][0][0][cindex][0][0] + alpha

        # generate edited image
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)

        # save tensor of edited image to png
        vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, str(lindex)+"_"+str(cindex)+"_"+image_name), padding=0)


def manyEdit_oneAttr_oneStyle(g_ema, image_name, latent_code_init, output_path, attr, lindex, cindex):
    """
    Many edit strength for specific attribute though (lindex, cindex) and save to output_path
    Args:
        g_ema:
        M:
        image_path:
        output_path:
        attr:
        lindex:
        cindex:

    Returns:

    """
    # file and dir preprocess
    print("You are working in:", image_name)
    output_path = os.path.join(output_path, image_name.split(".")[-2])
    os.makedirs(output_path, exist_ok=True)


    # Get S
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
    # latent = [s.detach().clone() for s in dlatents_loaded]

    count = 0
    # for alpha in np.arange(-15, 16, 2):
    for alpha in [-7, 9]:
        latent = [s.detach().clone() for s in dlatents_loaded]
        # edit specific (lindex, cindex) of stylespace
        latent[lindex][0][0][cindex][0][0] = latent[lindex][0][0][cindex][0][0] + alpha

        # generate edited image
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)

        # save tensor of edited image to png
        img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
        img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
        img_save.save(os.path.join(output_path, str(count)+".png"))

        # vutils.save_image(((img_gen + 1) / 2).clamp_(0, 1).data, os.path.join(output_path, str(count)+".png"), padding=0)
        count += 1

# def manyEdit_manyAttr_manyStyle(g_ema, M, image_path, output_path, attr, lindex, cindex):

def interpolation_latent(g_ema, M, image_path, output_path, attr, lindex, cindex, target_model):
    # file and dir preprocess
    image_name = image_path.split("/")[-1]
    print("You are working in:", image_name)
    os.makedirs(output_path, exist_ok=True)

    # Xception
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to("cuda:0")

    # set edit stregth
    alpha = 30  # hair

    # Get Latent, W
    latent_code_init = get_real_latent(image_path)
    if latent_code_init is None:
        return

    # Get S
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
    latent_original = [s.detach().clone() for s in dlatents_loaded]
    latent_edit = [s.detach().clone() for s in dlatents_loaded]

    # edit specific (lindex, cindex) of stylespace
    latent_edit[lindex][0][0][cindex][0][0] = latent_edit[lindex][0][0][cindex][0][0] + alpha

    beta = torch.tensor(0.5, requires_grad=True)
    # define optimizer
    optimizer = optim.Adam([beta], lr=0.1)
    trans = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    is_success = False
    final_img = torch.zeros(1, 3, 1024, 1024)
    first = True
    iter_count = 0
    for i in range(100):
        latent = []
        for i in range(len(latent_original)):
            latent.append(latent_original[i] * beta + latent_edit[i] * (1-beta))
        print(latent[lindex][0][0][cindex][0][0])
        # generate edited image
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)
        final_img = img_gen
        # vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, image_name), padding=0)
        before_softmax = target_model(trans((img_gen + 1) / 2))
        prob_list = F.softmax(before_softmax.squeeze())
        prob = prob_list[1]
        print(prob)
        iter_count += 1
        # TODO
        if prob < 0.1 and iter_count >= 20:
            vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, image_name),
                              padding=0)
            print("success", i)
            is_success = True
            break

        # TODO
        loss = criterion(before_softmax, torch.tensor([0]).cuda())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()

    # save tensor of edited image to png
    if not is_success:
        vutils.save_image(((final_img + 1) / 2).data, os.path.join(output_path, image_name), padding=0)

def interpolation_latent_v2(g_ema, M, image_path, output_path, config, target_model):
    # file and dir preprocess
    image_name = image_path.split("/")[-1]
    print("You are working in:", image_name)
    os.makedirs(output_path, exist_ok=True)

    # Xception
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to("cuda:0")

    # set edit stregth
    alpha = 30  # hair

    # Get Latent, W
    latent_code_init = get_real_latent(image_path)
    if latent_code_init is None:
        return

    # Get S
    with torch.no_grad():
        _, _, dlatents_loaded = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
    latent_original = [s.detach().clone() for s in dlatents_loaded]
    latent_edit = [s.detach().clone() for s in dlatents_loaded]

    # edit specific (lindex, cindex) of stylespace
    for key, value in config.items():
        lindex, cindex = value[0], value[1]
        latent_edit[lindex][0][0][cindex][0][0] = latent_edit[lindex][0][0][cindex][0][0] + alpha

    beta = torch.tensor(0.5, requires_grad=True)
    # define optimizer
    optimizer = optim.Adam([beta], lr=0.1)
    trans = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    is_success = False
    final_img = torch.zeros(1, 3, 1024, 1024)
    first = True
    iter_count = 0
    for i in range(100):
        latent = []
        for i in range(len(latent_original)):
            latent.append(latent_original[i] * beta + latent_edit[i] * (1-beta))
        for key, value in config.items():
            lindex, cindex = value[0], value[1]
            print(latent_edit[lindex][0][0][cindex][0][0])
        # generate edited image
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=True)
        final_img = img_gen
        # vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, image_name), padding=0)
        before_softmax = target_model(trans((img_gen + 1) / 2))
        prob_list = F.softmax(before_softmax.squeeze())
        prob = prob_list[1]
        print(prob)
        iter_count += 1
        # TODO
        if prob < 0.5 and iter_count >= 20:
            vutils.save_image(((img_gen + 1) / 2).data, os.path.join(output_path, image_name),
                              padding=0)
            print("success", i)
            is_success = True
            break

        # TODO
        loss = criterion(before_softmax, torch.tensor([0]).cuda())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()

    # save tensor of edited image to png
    if not is_success:
        vutils.save_image(((final_img + 1) / 2).data, os.path.join(output_path, image_name), padding=0)

if __name__ == '__main__':

    # 0.Define Generator
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    # mean_latent = g_ema.mean_latent(4096)

    # 1.Define manipulator method
    # M = Manipulator(dataset_name='ffhq')

    # 2. Define variable
    experiment_type = "ffhq_encode"
    os.chdir('/8T/work/search/Semantic/multi-semantic/global_directions/content/encoder4editing')
    EXPERIMENT_ARGS = {}
    EXPERIMENT_ARGS['transform'] = transforms.Compose([transforms.ToTensor()])
    dir_path = "/8T/work/search/dataset/study/test"
    output_path = "/8T/work/search/dataset/study/test/edit"
    attr = "wave hair"
    # lindex = 17
    # cindex = 91

    # 3. range all images
    for image in os.listdir(dir_path):
        ## 1. fine-tuning the latent code
        image_path = os.path.join(dir_path, image)
        original_image = Image.open(image_path)
        original_image = original_image.convert("RGB")
        if experiment_type == "ffhq_encode":
            input_image = run_alignment(image_path)
        else:
            input_image = original_image
        img_transforms = EXPERIMENT_ARGS['transform']
        transformed_image = img_transforms(input_image)

        # Get Latent, W
        latent_code_init = get_real_latent(image_path)

        delta = latent_code_init.clone()
        delta = delta.cuda()
        delta.requires_grad = True
        optimizer = optim.Adam([delta], lr=0.01)
        ori_im = transformed_image.cuda()
        for i in range(0):
            img_gen, _ = g_ema([delta], input_is_latent=True, randomize_noise=False, input_is_stylespace=False)
            gen_im = denorm1(torch.squeeze(img_gen, 0))
            loss = cal_loss.cal_loss(ori_im, gen_im, delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        latent_code_finetuned = delta.detach().clone()  # fine-tuned latent code

        ## 2. Edit
        # oneEdit_oneAttr_oneStyle(g_ema, M, image, latent_code_init, output_path, attr, lindex, cindex)
        # manyEdit_oneAttr_oneStyle(g_ema, M, image, latent_code_finetuned, output_path, attr, lindex, cindex)

        ## 3. Many style
        ### 3.1 Background region
        configs_ffhq_lighting = [[21, 16]]

        configs_ffhq_test = [[12, 479]]
        ### Eye region
        configs_ffhq_eyeball_size = [[20, 93], [18, 33], [18, 17], [17, 163]]

        configs_ffhq_eyeball_position = [[9, 409], [12, 43], [12, 149]]

        configs_ffhq_eyeball_color = [[20, 92], [20, 74]]

        configs_ffhq_eye_size = [[8, 78], [9, 167], [11, 87]]

        ### 3.2 Eyebrow region
        configs_ffhq_bushy_eyebrow = [[9, 440], [11, 290], [11, 433], [11, 364],
                                      [12, 100], [12, 102], [12, 242], [12, 278],
                                      [12, 315], [12, 325], [12, 455]]

        eyebrow = [[6, 228], [9, 440], [9, 454], [9, 510], [11, 35], [11, 290], [11, 433],
                                                [11, 364], [12, 64], [12, 100], [12, 102], [12, 123], [12, 166],
                                                [12, 242], [12, 278], [12, 312], [12, 315], [12, 325], [12, 443],
                                                [12, 455], [14, 2], [9, 30], [11, 364]]
        configs_ffhq_eyebrow_shape = [[14, 2], [8, 6], [8, 39], [8, 56],
                                       [8, 503], [9, 233], [9, 340], [9, 407],
                                       [11, 194], [11, 312], [11, 320], [9, 30]]

        ### 3.3 Skin region
        configs_ffhq_pale_skin = [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
                      [15, 102]]

        configs_ffhq_dark_circles = [[14, 79], [14, 194], [14, 5]]


        ### 3.4 Ear region
        configs_ffhq_ear = [[8, 81], [11, 15], [15, 47]]

        ### 3.5 Hair region
        configs_ffhq_bangs = [[6, 487], [6, 322], [3, 259], [6, 285], [5, 414],  [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45],
                              [6, 208], [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204], [6, 313], [6, 343], [6, 413], [15, 97],
                              [9, 18], [9, 118], [9, 130]]

        configs_ffhq_wave = [[5, 92], [6, 323], [6, 394], [6, 500], [8, 128]]

        configs_ffhq_blond_hair = [[12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [3, 486], [17, 249], [17, 92]]

        configs_ffhq_gray_hair = [[14, 4], [12, 287], [11, 286], [17, 19], [15, 191]]

        ### 3.6 Mouth region

        configs_ffhq_lipstick = [[11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35]]
        configs_ffhq_lipstick_all = [[20, 73], [15, 251], [15, 249], [15, 228], [15, 178], [15, 121], [15, 75], [15, 68], [14, 490], [14, 263], [14, 230], [14, 223], [14, 222], [14, 213], [14, 110], [14, 107], [14, 85], [14, 66], [14, 12], [12, 482], [12, 410], [12, 253], [12, 239], [12, 186], [12, 183], [12, 177], [12, 80], [12, 59], [12, 56], [11, 447], [11, 374], [11, 314], [11, 204], [11, 174], [11, 86], [9, 321], [9, 294], [9, 232],[11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [8, 191]]

        configs_ffhq_mouth = [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232],
                              [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
                              [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]]
        configs_ffhq_mouth_all = [[12, 262], [8, 17], [6, 501], [6, 491], [6, 378], [6, 202], [6, 113], [6, 21], [17, 241], [17, 112], [9, 91], [9, 117], [9, 351], [9, 77], [17, 186], [11, 279], [9, 452], [9, 48], [9, 26], [8, 456], [8, 389], [8, 122], [8, 118], [8, 85], [18, 0], [18, 52], [18, 57], [6, 259], [6, 214], [11, 313], [11, 409], [14, 286], [15, 104], [17, 29], [17, 37], [17, 126], [11, 481], [17, 165]]
        configs_ffhq_hair = [[12, 479], [12, 298], [12, 330], [12, 390], [12, 424], [11, 125], [6, 137], [14, 27], [14, 55], [9, 38], [14, 146], [3, 337], [3, 302], [14, 186], [3, 296], [14, 225], [9, 33], [14, 289], [3, 486], [9, 84], [14, 308], [12, 287], [5, 228], [11, 337], [11, 376], [11, 406], [11, 420], [5, 152], [9, 183], [5, 111], [12, 119], [5, 57], [5, 263], [12, 236], [12, 238], [12, 249], [12, 266], [14, 294], [9, 208], [9, 370], [17, 19], [17, 92], [17, 149], [17, 249], [8, 192], [21, 11], [8, 4], [23, 15], [9, 371], [23, 56], [23, 58], [8, 339], [15, 191], [14, 417], [14, 426], [14, 470], [5, 397], [15, 175], [15, 62], [15, 97], [15, 162], [11, 34], [11, 262], [6, 204], [6, 262], [6, 258], [24, 12], [6, 306], [8, 75], [8, 45], [8, 30], [8, 25], [6, 510], [6, 496], [6, 490], [6, 482], [6, 480], [6, 413], [6, 313], [6, 405], [6, 404], [6, 397], [6, 343], [6, 337], [6, 331], [6, 152], [3, 98], [3, 233], [3, 217], [3, 173], [3, 131], [3, 104], [3, 4], [3, 355], [2, 499], [2, 364], [2, 216], [2, 187], [2, 84], [2, 70], [2, 43], [2, 27], [0, 224], [0, 51], [0, 49], [3, 253], [3, 381], [6, 81], [5, 248], [6, 67], [6, 45], [5, 502], [5, 491], [5, 457], [5, 446], [5, 409], [5, 362], [5, 238], [3, 384], [5, 212], [5, 186], [5, 176], [5, 166], [8, 88], [5, 131], [5, 79], [5, 44], [5, 10], [8, 80], [9, 49], [8, 89], [11, 214], [14, 219], [14, 172], [14, 149], [14, 122], [14, 28], [14, 4], [12, 463], [12, 456], [12, 434], [12, 401], [12, 375], [12, 314], [12, 295], [12, 279], [12, 226], [12, 206], [12, 192], [12, 120], [12, 117], [12, 94], [11, 497], [11, 470], [11, 461], [11, 303], [11, 286], [11, 260], [11, 254], [11, 232], [14, 236], [14, 329], [17, 6], [23, 60], [23, 34], [23, 30], [23, 29], [21, 18], [20, 108], [20, 33], [20, 3], [18, 112], [18, 84], [18, 77], [17, 48], [17, 9], [15, 247], [14, 348], [15, 205], [15, 166], [15, 163], [15, 138], [15, 122], [15, 106], [15, 96], [15, 42], [14, 499], [14, 497], [14, 452], [14, 419], [14, 390], [11, 226], [11, 180], [8, 138], [11, 131], [9, 126], [9, 114], [9, 112], [9, 99], [9, 83], [9, 64], [23, 61], [9, 25], [8, 497], [8, 432], [8, 380], [8, 365], [8, 347], [8, 344], [8, 341], [8, 322], [8, 321], [8, 287], [8, 240], [8, 231], [8, 219], [8, 212], [8, 199], [8, 178], [8, 172], [8, 165], [9, 161], [9, 203], [9, 436], [11, 130], [11, 89], [11, 85], [11, 83], [11, 69], [11, 59], [11, 52], [11, 45], [11, 30], [11, 9], [9, 492], [9, 430], [9, 212], [9, 422], [9, 388], [9, 385], [9, 377], [9, 349], [9, 302], [9, 261], [9, 257], [0, 45]]

        configs_ffhq_glasses = [[2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228]]
        # many times
        for value in configs_ffhq_test:
            lindex, cindex = value[0], value[1]
            output_path_tmp = os.path.join(output_path, str(lindex)+"_"+str(cindex))
            manyEdit_oneAttr_oneStyle(g_ema, image, latent_code_finetuned, output_path_tmp, attr, lindex, cindex)

        # oneEdit_manyAttr_manyStyle(g_ema, M, image, latent_code_init, output_path, configs_ffhq_ear)
        # break
