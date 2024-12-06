# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : all_attribute_attack.py
# @Project     : multi-semantic
# @CreateTime  : 2022/4/4
# H9: 05:10
# @Version     : v1.0
# @Description : ""
# @Update      : [��][�YYYY-MM-DD] [�9��
# ][����]
# @Copyright � 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
import argparse
import math
import os
import sys
import torch
import cal_loss
import torchvision
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import cv2
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
from matplotlib import pyplot as plt
from utils.utils import ensure_checkpoint_exists
from global_directions.manipulate import Manipulator
from torchvision import transforms
from GetCode_real import get_real_latent
from MapTS import GetFs, GetBoundary, GetDt
from torchvision import transforms
from global_directions.content.encoder4editing.criteria.id_loss import IDLoss
from models.stylegan2.model import Generator
from util_target_model import FFD
from util_target_model import CNNDetection
from util_target_model import GramNet
from util_target_model import F3_Net
from util_target_model import Xception
from util_target_model import Efficientnetb7
from util_target_model import patch_forensics
from util_target_model import ResNet
from global_directions.content.encoder4editing.models.discriminator import LatentCodesDiscriminator

f = open("/8T/xiangtao/new/dataset/attribute/ff++/impact/patch-forensics/output.txt", "w+")

def cal_dis_loss(latent, first_dis, loss_dis_init):
    loss_disc = 0
    s_list = []
    for i in range(15):
        s_list.append(latent[i])
    ## deal with special situation
    test1 = torch.cat([latent[15], latent[16]], 2)
    s_list.append(test1)
    test2 = torch.cat([latent[17], latent[18], latent[19]], 2)
    s_list.append(test2)
    xx = torch.zeros_like(latent[20])
    test3 = torch.cat(
        [latent[20], latent[21], latent[22], latent[23], latent[24],
         latent[25], xx], 2)
    s_list.append(test3)
    # transform list to tensor(N, 18, 512
    fake = []
    for i in range(len(s_list)):
        fake.append(s_list[i][:, 0, :, 0, 0])
    fake_s_all = torch.stack(fake, 1)
    fake_pred = discriminator(fake_s_all)
    loss_disc += F.softplus(fake_pred).mean()
    if first_dis:
        loss = loss_disc - loss_disc
    else:
        loss = loss_disc - loss_dis_init
        if loss.item() < 0:
            loss = loss_disc - loss_disc
    return loss * 1000, loss_disc

def cal_id_loss(y_hat, y, x):
    loss_id, sim_improvement, id_logs = id_loss(y_hat, y.unsqueeze(0).cuda(), x.unsqueeze(0).cuda())

    return loss_id

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


# Attack
def main(nothing, quality, all, image_name, output_path, output_inversion_path, configs_all_semantic, latent_code_init, noise, model,
         loss_func, target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init):
    try:
        if nothing:
            # define working file
            image_name = image_path.split("/")[-1]
            print("You are working in:", image_name)
            noise_in = None
        else:
            noise_in = []
            for key, value in noise.items():
                noise_in.append(value.detach().cuda())


        latent_code_finetuned = latent_code_init.cuda()
        if latent_code_init is None:
            return
        with torch.no_grad():
            _, _, dlatents_loaded = g_ema([latent_code_finetuned], input_is_latent=True, return_latents=True,
                                          randomize_noise=False, noise=noise_in)#noise_in
        latent = [s.detach().clone() for s in dlatents_loaded]

        # Define our stylespace channels that are edited
        alpha = []
        count = 0
        for config in configs_all_semantic:
            lindex = config[0]
            cindex = config[1]
            if lindex == 2100 and cindex == 1600:  # pale_skin
                alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy()))
            #
            # elif lindex == 6 and cindex == 228: #glasses
            #     alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy() - 25))
            #
            # elif lindex == 5 and cindex == 85: #bangs
            #     alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy() + 15))

            # print(latent[lindex][0][0][cindex].item())
            else:
                alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy()))

            alpha[count] = alpha[count].cuda()
            alpha[count].requires_grad = True
            count += 1

        # define button
        is_success = False
        if target_model_name in ["GramNet"]:
            final_img = torch.zeros(1, 3, 512, 512)
        elif target_model_name in ["patch_forensics"]:
            final_img = torch.zeros(1, 3, 128, 128)
        else:
            final_img = torch.zeros(1, 3, 299, 299)
        trans_save = transforms.Compose([transforms.ToPILImage()])
        first = True
        min_fake_score = 1
        last_fake_score = 1
        pre_fake_score = 1
        is_stopping = False
        ## TODO

        attack_count = 0
        if all:
            for j in range(len(attribute_channel)):

                first_pred = True
                edit_channel = attribute_channel[j]
                edit_lindex = edit_channel[0]
                edit_cindex = edit_channel[1]
                edit_strength = attribute_edit[j]
                # print("change channel", edit_strength)

                if j == 0:
                    count = 0
                    for config in configs_all_semantic:
                        lindex = config[0]
                        cindex = config[1]
                        latent[lindex][0][0][cindex] = alpha[count]
                        count += 1
                else:
                    count = 0
                    for config in configs_all_semantic:
                        lindex = config[0]
                        cindex = config[1]
                        if lindex == edit_lindex and cindex == edit_cindex:
                            alpha[count].data = (alpha[count].data / alpha[count].data) * edit_strength
                        count += 1

                for lr in lr_list:
                    # print("you are using :", lr)
                    # lambda_dis = lr
                    # lambda_id = lr
                    stop_count = 0
                    final_stop = 0
                    count_times = 0
                    # define optimizer
                    optimizer2 = optim.Adam(alpha, lr=lr)
                    ## range
                    for i in range(400):  ##TODO
                        # print(final_stop)
                        # print(stop_count)

                        if stop_count > 200 or final_stop > 200:  # TODO
                            break
                        else:
                            count = 0
                            for config in configs_all_semantic:
                                lindex = config[0]
                                cindex = config[1]
                                latent[lindex][0][0][cindex] = alpha[count]
                                count += 1

                        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, noise=noise_in,#noise_in
                                           input_is_stylespace=True)
                        if target_model_name in ["GramNet"]:
                            img_tensor = transforms.Resize((512, 512))(((img_gen + 1) / 2).clamp_(0, 1))
                        elif target_model_name in ["patch_forensics"]:
                            img_tensor = transforms.Resize((128, 128))(((img_gen + 1) / 2).clamp_(0, 1))
                        else:
                            img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))

                        # img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
                        # img_save.save(os.path.join(output_path, image_name))

                        if first:
                            first = False

                        # Adv loss
                        ## 1. FFD
                        if target_model_name is "FFD":
                            loss_adv, fake_score = FFD(img_tensor, model, loss_func)
                        ## 2. CNNDetection
                        elif target_model_name is "CNNDetection":
                            loss_adv, fake_score = CNNDetection(img_tensor, model, loss_func)
                        ## 3. GramNet
                        elif target_model_name is "GramNet":
                            loss_adv, fake_score = GramNet(img_tensor * 255, model, loss_func)
                        ## 4. F3-Net
                        elif target_model_name is "F3_Net":
                            loss_adv, fake_score = F3_Net(img_tensor, model, loss_func)
                        elif target_model_name is "Xception":
                            loss_adv, fake_score = Xception(img_tensor, model, loss_func)
                        elif target_model_name is "patch_forensics":
                            loss_adv, fake_score = patch_forensics(img_tensor, model, loss_func)
                        elif target_model_name is "ResNet":
                            loss_adv, fake_score = ResNet(img_tensor, model, loss_func)
                        ## 5. Efficientnetb7
                        # loss, fake_score = Efficientnetb7(((img_gen + 1) / 2).clamp_(0, 1))
                        # print(fake_score.item())
                        # print(loss.item())
                        if j != 0 and first_pred:
                            if pre_fake_score <= fake_score.item():
                                # print(j, "no use")
                                break
                            else:
                                first_pred = False



                        if nothing:
                            loss = loss_adv
                        elif all or quality:
                            # distribution loss
                            if first_dis:
                                loss_dis, loss_score = cal_dis_loss(latent, first_dis, loss_dis_init)
                                loss_dis_init = loss_score.item()
                                first_dis = False
                            else:
                                loss_dis, loss_score = cal_dis_loss(latent, first_dis, loss_dis_init)

                            # id loss
                            loss_id = cal_id_loss(img_tensor, original_image, original_image)

                            loss = loss_adv + lambda_dis * loss_dis + lambda_id * loss_id

                        ## save the min fake score
                        if count_times % 10 == 0:
                            if (last_fake_score - fake_score.item()) < 0.01:
                                final_stop += 1
                            elif (last_fake_score - fake_score.item()) > 0.05:
                                if final_stop > 0:
                                    final_stop -= 1
                            last_fake_score = fake_score.item()

                        if fake_score.item() <= min_fake_score:
                            final_img = img_tensor
                            min_fake_score = fake_score.item()

                        if fake_score.item() >= pre_fake_score:
                            stop_count += 1
                        else:
                            stop_count = 0

                        # print("stop", stop_count)
                        # print("final", final_stop)
                        pre_fake_score = fake_score.item()
                        count_times += 1
                        ## stop condition
                        print(fake_score.item())
                        if fake_score.item() < 0.4: #0.48
                            fail_count = 0
                            is_success = True
                            img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
                            xx = np.array(img_save)
                            img_save.save(os.path.join(output_path, image_name))
                            break
                        optimizer2.zero_grad()
                        loss_adv.backward(retain_graph=True)
                        optimizer2.step()
                        attack_count += 1
                    if is_success:
                        break
                if is_success:
                    break
        else:
            for lr in lr_list:
                # print("you are using :", lr)
                # lambda_dis = lr
                # lambda_id = lr
                stop_count = 0
                final_stop = 0
                count_times = 0
                # define optimizer
                optimizer2 = optim.Adam(alpha, lr=lr)
                ## range
                for i in range(600):  ##TODO
                    # print(final_stop)
                    # print(stop_count)

                    if stop_count > 20 or final_stop > 30:  # TODO
                        break
                    else:
                        count = 0
                        for config in configs_all_semantic:
                            lindex = config[0]
                            cindex = config[1]
                            latent[lindex][0][0][cindex] = alpha[count]
                            count += 1

                    img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, noise=noise_in,
                                       input_is_stylespace=True)
                    if target_model_name in ["GramNet", "patch_forensics"]:
                        img_tensor = transforms.Resize((512, 512))(((img_gen + 1) / 2).clamp_(0, 1))
                    else:
                        img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))

                    if first:
                        # img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
                        # img_save.save(os.path.join(output_inversion_path, image_name))
                        first = False
                    # img_tensor = img_tensor.type(torch.CharTensor)

                    # Adv loss
                    ## 1. FFD
                    if target_model_name is "FFD":
                        loss_adv, fake_score = FFD(img_tensor, model, loss_func)
                    ## 2. CNNDetection
                    elif target_model_name is "CNNDetection":
                        loss_adv, fake_score = CNNDetection(img_tensor, model, loss_func)
                    ## 3. GramNet
                    elif target_model_name is "GramNet":
                        loss_adv, fake_score = GramNet(img_tensor * 255, model, loss_func)
                    ## 4. F3-Net
                    elif target_model_name is "F3_Net":
                        loss_adv, fake_score = F3_Net(img_tensor, model, loss_func)
                    elif target_model_name is "Xception":
                        loss_adv, fake_score = Xception(img_tensor, model, loss_func)
                    elif target_model_name is "patch_forensics":
                        loss_adv, fake_score = patch_forensics(img_tensor, model, loss_func)
                    ## 5. Efficientnetb7
                    # loss, fake_score = Efficientnetb7(((img_gen + 1) / 2).clamp_(0, 1))
                    print(fake_score.item())

                    if nothing:
                        loss = loss_adv
                    elif all or quality:
                        # distribution loss
                        loss_dis = cal_dis_loss(latent, first_dis, loss_dis_init)

                        # id loss
                        loss_id = cal_id_loss(img_tensor, original_image, original_image)

                        loss = loss_adv + lambda_dis * loss_dis + lambda_id * loss_id

                    ## save the min fake score
                    if count_times % 10 == 0:
                        if (last_fake_score - fake_score.item()) < 0.01:
                            final_stop += 1
                        elif (last_fake_score - fake_score.item()) > 0.05:
                            if final_stop > 0:
                                final_stop -= 1
                        last_fake_score = fake_score.item()

                    if fake_score.item() <= min_fake_score:
                        final_img = img_tensor
                        min_fake_score = fake_score.item()

                    if fake_score.item() >= pre_fake_score:
                        stop_count += 1
                    else:
                        stop_count = 0

                    print("stop", stop_count)
                    print("final", final_stop)
                    pre_fake_score = fake_score.item()
                    count_times += 1
                    ## stop condition
                    # print(fake_score.item())
                    if fake_score.item() < 0.1:
                        fail_count = 0
                        is_success = True
                        img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
                        img_save.save(os.path.join(output_path, image_name))
                        break
                    optimizer2.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer2.step()
                    attack_count += 1
                if is_success:
                    break

        # if not success, store final edit
        if not is_success:
            img_save = transforms.ToPILImage()(final_img.squeeze(0))
            img_save.save(os.path.join(output_path, image_name))
            # print(image_name)
            fail_count = 1
            print("fail!!!")
        print("attack count:", attack_count)
        return fail_count
    except TypeError:
        print("exception !!!!!!!!!!!!!")


if __name__ == '__main__':
    """
        **************************************                                  
        0x00 Define parameters
        **************************************
    """
    # dir_path = "/8T/xiangtao/old/dataset/test_ff++"
    dir_path = "/8T/xiangtao/new/dataset/original_1024_100_299/test"
    latent_path = "/8T/xiangtao/new/dataset/inversion/stylegan/latents/with_noise_100"
    noise_path = "/8T/xiangtao/new/dataset/inversion/stylegan/noise/with_noise_100"
    output_path = "/8T/xiangtao/new/dataset/adversarial_training/"
    output_inversion_path = "/8T/xiangtao/new/dataset"
    nothing = False  # without quality and optimization
    quality = False #with quality and without optimization
    all = True # with quality and optimization

    target_model_name = "patch_forensics"
    lambda_dis = 0.001
    lambda_id = 0.001
    ##TODO
    # lambda_dis = 0
    # lambda_id = 0
    attribute_channel = [[9, 232]]
    attribute_edit = [0]
    if all:
        lr_list = [0.1]
    else:
        lr_list = [0.1]


    """
        **************************************                                  
        0x01 Define and Load Models
        **************************************
    """
    # 1. StyleGAN2's Generator
    g_ema = Generator(1024, 512, 8)
    weights = torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"]
    g_ema.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    # 2. discriminator
    discriminator = LatentCodesDiscriminator(512, 4).cuda()
    ckpt = torch.load("/8T/xiangtao/new/weights/best_model.pt",
                      map_location='cpu')
    discriminator.load_state_dict(ckpt['discriminator_state_dict'])
    discriminator.cuda()

    # 3. id
    id_loss = IDLoss().cuda().eval()

    # 4. Target model
    if target_model_name is "F3_Net":
        # 0. Import
        from target_model.F3net.trainer import Trainer

        # 1. Define and Load Model
        # config
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        pretrained_path = 'pretrained/xception-b5690688.pth'
        gpu_ids = [*range(osenvs)]
        model = Trainer(gpu_ids, "Both",
                        '/8T/xiangtao/new/code/multi-semantic/global_directions/pretrained/xception-b5690688.pth')
        model.load('/8T/xiangtao/new/code/multi-semantic/weights/F3-Net/2_500_best.pkl')
        model.model.eval()

        # 2. Define Loss
        loss_func = nn.BCEWithLogitsLoss().cuda()
        #################
    elif target_model_name is "CNNDetection":
        # 0. Import
        from target_model.CNNDetection.networks.resnet import resnet50

        # 1. Define and Load Model
        model = resnet50(num_classes=1)
        state_dict = torch.load("/8T/xiangtao/new/code/multi-semantic/weights/CNNDetection/blur_jpg_prob0.1.pth", map_location='cpu')
        # state_dict = torch.load("/8T/work/search/Semantic/CNNDetection-master/checkpoints/blur_jpg_prob0.1_latest/model_epoch_best.pth", map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.eval()
        model.cuda()

        # 2. Define Loss
        loss_func = nn.BCEWithLogitsLoss().cuda()
    elif target_model_name is "GramNet":
        # 0. Import
        import resnet18_gram as resnet

        # 1. Define and Load model
        # model = resnet.resnet18(pretrained=True)  # resnet18_gram.resnet18() #pretrained=True)
        model = torch.load('/8T/xiangtao/new/code/multi-semantic/target_model/GramNet/weights/stylegan-ffhq.pth')
        model.eval()
        model.cuda()

        # 2. Define Loss
        # loss_func = nn.CrossEntropyLoss().cuda()
        loss_func = nn.BCEWithLogitsLoss().cuda()
    elif target_model_name is "FFD":
        # Import
        from target_model.FFD.network.xception import Model
        from target_model.FFD.network.templates import get_templates

        # Define
        ## 0. loss
        loss_func = nn.CrossEntropyLoss().cuda()
        # loss_func = nn.BCEWithLogitsLoss().cuda()

        ## 1. model
        BACKBONE = 'xcp'
        MAPTYPE = 'tmp'
        TEMPLATES = None
        MODEL_DIR = "/8T/xiangtao/new/code/multi-semantic/target_model/FFD/pretrained model/xcp_tmp"
        if MAPTYPE in ['tmp', 'pca_tmp']:
            TEMPLATES = get_templates()
        model = Model(MAPTYPE, TEMPLATES, 2, False)
        model.load(75, MODEL_DIR)
        model.model.eval()
        model.model.cuda()
    elif target_model_name is "Xception":
        # 0. Import
        from network.models import model_selection

        # 1. Define and Load Model
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
        model = torch.load("/8T/xiangtao/new/code/multi-semantic/weights/weights/xception/full_raw.p")
        model.eval()
        model.cuda()

        # 2. Define Loss
        loss_func = nn.CrossEntropyLoss().cuda()
    elif target_model_name is "ResNet":
        from target_model.CNNDetection.networks.resnet import resnet50
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2)
        model = model.cuda()
        checkpoint = torch.load("/8T/xiangtao/new/code/Pytorch-Adversarial-Training/checkpoint/basic/basic_training55")
        model.load_state_dict(checkpoint['model'])
        model.eval()

        loss_func = nn.BCEWithLogitsLoss().cuda()
    elif target_model_name is "patch_forensics":
        # 0. Import
        import yaml
        from target_model.patch_forensics.options.test_options import TestOptions
        from target_model.patch_forensics.models import create_model


        # 1. Define and Load Model
        opt = TestOptions().parse()
        # opt.gpu_ids = [0]
        opt.which_epoch = "bestval"
        # opt.partition = "test"
        opt.checkpoints_dir = "/8T/xiangtao/new/code/patch-forensics/checkpoints"
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        # 2. Define Loss
        loss_func = nn.CrossEntropyLoss().cuda()
        # loss_func = nn.BCEWithLogitsLoss().cuda()

    """
        **************************************                                  
        0x02 Attack
        **************************************
    """
    ####
    # 1. Individual attributes
    ####
    # configs_all_attributes = {
    #     # "pale_skin": [[24, 12]],
    #     "pale_skin": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102]],
    #     "hairstyle": [[5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                   [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                   [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                   [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130]],
    #     "hair_color": [[12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                    [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191]],
    #     "lipstick": [[11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235],
    #                  [17, 66], [17, 247], [18, 35]],
    #     "opend_mouth": [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                     [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
    #                     [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]],
    #     "bushy_eyebrow": [[9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
    #                       [12, 315], [12, 325], [12, 455]],
    #     "eyebrow_shape": [[14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194],
    #                       [11, 312], [11, 320], [9, 30]],
    #     "earring": [[8, 81], [11, 15], [15, 47]],
    #     "eyeball_position": [[9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                          [9, 510], [11, 35]],
    #     "eye_size": [[8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315]],
    #     "galsses": [[2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228]]
    #     }

    ####
    # 2. Impact attribute combinations
    ####
    configs_all_attributes = {
        "earring": [[8, 81], [11, 15], [15, 47]],
        # "lipstick": [[11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35]],
        "earring+glasses": [[8, 81], [11, 15], [15, 47], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228]],
        "earring+glasses+eye_close": [[8, 81], [11, 15], [15, 47], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315]],
        "earring+glasses+eye_close+eyebrow_shape": [[8, 81], [11, 15], [15, 47], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312], [11, 320], [9, 30]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow": [[8, 81], [11, 15], [15, 47], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312], [11, 320], [9, 30], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick": [[8, 81], [11, 15], [15, 47], [2, 175], [2, 97],
                                                                  [3, 120], [5, 325], [3, 288], [6, 228], [8, 78],
                                                                  [9, 167], [11, 87], [9, 454], [12, 64], [12, 315],
                                                                  [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233],
                                                                  [9, 340], [9, 407], [11, 194], [11, 312], [11, 320],
                                                                  [9, 30], [9, 440], [11, 290], [11, 433], [11, 364],
                                                                  [12, 100], [12, 102], [12, 242], [12, 278], [12, 315],
                                                                  [12, 325], [12, 455], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick+eyeball_position": [[8, 81], [11, 15], [15, 47], [2, 175],
                                                                           [2, 97], [3, 120], [5, 325], [3, 288], [6, 228],
                                                                           [8, 78], [9, 167], [11, 87], [9, 454], [12, 64],
                                                                           [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503],
                                                                           [9, 233], [9, 340], [9, 407], [11, 194], [11, 312],
                                                                           [11, 320], [9, 30], [9, 440], [11, 290], [11, 433],
                                                                           [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                                                                           [12, 315], [12, 325], [12, 455], [11, 6], [11, 73],
                                                                           [11, 116], [12, 5], [14, 110], [14, 490],
                                                                           [15, 37], [15, 45], [15, 235], [17, 66],
                                                                           [17, 247], [18, 35], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick+eyeball_position+opend_mouth": [[8, 81], [11, 15], [15, 47], [2, 175],
                                                                           [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64],
                                                                           [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312],
                                                                           [11, 320], [9, 30], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                                                                           [12, 315], [12, 325], [12, 455], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66],[17, 247], [18, 35], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
                                                                           [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick+eyeball_position+opend_mouth+hairstyle": [[8, 81], [11, 15], [15, 47], [2, 175],
                                                                                                        [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64],
                                                                                                        [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312],
                                                                                                        [11, 320], [9, 30], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                                                                                                        [12, 315], [12, 325], [12, 455], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279],
                                                                                                        [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414], [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208], [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204], [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick+eyeball_position+opend_mouth+hairstyle+hair_color": [[8, 81], [11, 15], [15, 47], [2, 175],
                                                                                                                  [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64],
                                                                                                                  [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312],
                                                                                                                  [11, 320], [9, 30], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                                                                                                                  [12, 315], [12, 325], [12, 455], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204],
                                                                                                                  [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414], [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208], [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111],
                                                                                                                  [5, 397], [6, 137], [6, 204], [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191]],
        "earring+glasses+eye_close+eyebrow_shape+bushy_eyebrow+lipstick+eyeball_position+opend_mouth+hairstyle+hair_color+pale_skin": [[8, 81], [11, 15], [15, 47], [2, 175],
                                                                                                                                       [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64],
                                                                                                                                       [12, 315], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312],
                                                                                                                                       [11, 320], [9, 30], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                                                                                                                                       [12, 315], [12, 325], [12, 455], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232],
                                                                                                                                       [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414], [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208], [6, 216],
                                                                                                                                       [6, 341], [6, 497],
                                                                                                                                       [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204], [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [21, 16], [14, 57], [12, 235], [12, 44], [14, 79],
                                                                                                                                       [14, 194], [14, 5], [14, 339], [24, 12], [15, 102]],
    }

    ####
    # 3. Variation attribute combinations
    ####
    # configs_all_attributes = {
    #     "opend_mouth": [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]],
    #     "opend_mouth+eyeclose": [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315]],
    #     "opend_mouth+eyeclose+eyeball_position": [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring": [[8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow": [[9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510],
    #                                                                     [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick": [[17, 66], [17, 247], [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87],
    #                                                                     [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick+galsses": [[2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [17, 66], [17, 247], [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26],
    #                                                                              [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick+galsses+eyebrow_shape": [[11, 312], [11, 320], [9, 30], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [17, 66], [17, 247], [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91],
    #                                                                                      [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick+galsses+eyebrow_shape+hair_color": [[3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [11, 312], [11, 320], [9, 30], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [17, 66], [17, 247], [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315],
    #                                                                                                    [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick+galsses+eyebrow_shape+hair_color+hairstyle": [[6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204], [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [11, 312], [11, 320], [9, 30], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [17, 66], [17, 247],
    #                                                                                                               [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315],
    #                                                                                                               [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409], [12, 43],
    #                                                                                                               [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],  [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208]],
    #     "opend_mouth+eyeclose+eyeball_position+earring+bushy_eyebrow+lipstick+galsses+eyebrow_shape+hair_color+hairstyle+pale_skin": [[15, 102], [21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12], [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204], [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302], [11, 312], [11, 320], [9, 30],
    #                                                                                                                         [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228], [17, 66], [17, 247],
    #                                                                                                                         [18, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315],
    #                                                                                                                         [12, 325], [12, 455], [8, 81], [11, 15], [15, 47], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117], [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [9, 409],
    #                                                                                                                         [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74], [9, 510], [11, 35], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414], [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208]],
    # }

    #####
    # original
    #####
    # configs_all_attributes = {
    #     "pale_skin": [[14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102]], #[21, 16],
    #     "pale_skin+hair_color": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                    [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191]],
    #
    #     "pale_skin+hair_color+hairstyle": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                    [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                   [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                   [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                   [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                    [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                   [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                   [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                   [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                     [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
    #                     [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                   [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                    [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                   [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                   [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                   [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                     [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
    #                     [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                          [9, 510], [11, 35]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                                                                     [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                                                                     [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                                                                     [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                                                                     [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                                                                     [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                                                                     [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
    #                                                                     [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                                                                     [9, 510], [11, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick+bushy_eyebrow": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                                                                              [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                                                                              [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                                                                              [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                                                                              [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                                                                              [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                                                                              [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
    #                                                                              [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                                                                              [9, 510], [11, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick+bushy_eyebrow+eyebrow_shape": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                                                                                            [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                                                                                            [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                                                                                            [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                                                                                            [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                                                                                            [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                                                                                            [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312], [11, 320], [9, 30],
    #                                                                                            [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                                                                                            [9, 510], [11, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick+bushy_eyebrow+eyebrow_shape+eye_size": [[8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                                                                                                          [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                                                                                                          [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                                                                                                          [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                                                                                                          [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                                                                                                          [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                                                                                                          [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312], [11, 320], [9, 30],
    #                                                                                                          [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                                                                                                          [9, 510], [11, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455]],
    #     "pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick+bushy_eyebrow+eyebrow_shape+eye_size+earring": [[8, 81], [11, 15], [15, 47], [8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315], [21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339], [24, 12],
    #                                                                                                                   [15, 102], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
    #                                                                                                                   [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
    #                                                                                                                   [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
    #                                                                                                                   [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
    #                                                                                                                   [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
    #                                                                                                                   [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118], [14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194], [11, 312], [11, 320], [9, 30],
    #                                                                                                                   [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
    #                                                                                                                   [9, 510], [11, 35], [11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235], [17, 66], [17, 247], [18, 35], [9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278], [12, 315], [12, 325], [12, 455]],
    #
    # }


    for key, value in configs_all_attributes.items():
        # if key not in ["pale_skin+hair_color+hairstyle+opend_mouth+eyeball_position+lipstick+bushy_eyebrow+eyebrow_shape+eye_size+earring"]:
        #     continue
        # 1. create folders
        tmp_output_path = os.path.join(output_path, key)
        # tmp_output_path = "/8T/xiangtao/new/dataset/adversarial_training/our_v2/all-all"
        print("Attribute:", key, file=f)
        print("Attribute:", key)
        configs_all_semantic = value
        if not os.path.exists(tmp_output_path):
            os.makedirs(tmp_output_path)
        if not os.path.exists(output_inversion_path):
            os.makedirs(output_inversion_path)

        # 2. Get latent from image or get latent directly
        if nothing:
            for image in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image)
                ## 3. get images's latent code
                latent_code = get_real_latent(image_path)
                noise = None
                original_image = None
                first_dis = True
                loss_dis_init = 0
                fail_count = main(nothing, quality, all, image, tmp_output_path, output_inversion_path, configs_all_semantic,
                                  latent_code, noise, model, loss_func, target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init)
        else:
            fail_all_count = 0 # count the number of fail attack
            for image in os.listdir(latent_path):
                image_name = image.split(".")[-2] + ".png"
                # 1. get y for id loss
                if image_name not in os.listdir(dir_path):
                    continue
                # if image_name not in ["004426.png"]:
                #     continue
                original_image = Image.open(os.path.join(dir_path, image_name))
                original_image = original_image.convert("RGB")
                original_image = transforms.ToTensor()(original_image)

                # 2. If the sample has already been attack, skip it
                if image_name in os.listdir(tmp_output_path):
                    continue
                # 3. get latent code
                print("You are workong :", image_name)
                latent_code = np.load(os.path.join(latent_path, image))
                latent_code = torch.from_numpy(latent_code)

                # 4. get noise
                noise = torch.load(os.path.join(noise_path, image.split(".")[-2] + ".pt"), map_location=torch.device('cpu'))

                # 5. attack
                first_dis = True
                loss_dis_init = 0

                fail_count = main(nothing, quality, all, image_name, tmp_output_path, output_inversion_path, configs_all_semantic,
                                  latent_code, noise,  model, loss_func, target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init)
                fail_all_count += fail_count
            print("fail_count:", fail_all_count, file=f)
            print("fail_count:", fail_all_count)