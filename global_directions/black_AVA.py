# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : attribute_black_attack.py
# @Project     : multi-semantic
# @CreateTime  : 2022/10/20 下午3:21:23
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
import cal_loss
import torchvision
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
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
from global_directions.content.encoder4editing.models.discriminator import LatentCodesDiscriminator
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.faceid.v20180301 import faceid_client, models
import base64
from tencentcloud.common.abstract_model import AbstractModel
from selenium import webdriver
from selenium.webdriver import DesiredCapabilities
from target_model.Baidu.aip import AipFace
import json
import time
import os
import requests
import glob
import pickle
import argparse
import pandas as pd
from shutil import copyfile

# f = open("/8T/work/search/dataset/study/StyleGAN/results/CNNDetection_all.txt", "w+")
class  DeepfakeRequest(AbstractModel):
    """Liveness请求参数结构体

    """
    def __init__(self):

        self.Action = None
        self.Version = None
        self.ImageBase64 = None
        self.VideoBase64 = None
        self.Limit = None

    def _deserialize(self, params):
        self.ImageBase64 = params.get("ImageBase64")
        self.Action = params.get("Action")

        self.Version = params.get("Version")
        self.Limit = params.get("Limit")

class DeepfakeResponse(AbstractModel):
    """LivenessCompare返回参数结构体

    """

    def __init__(self):
        """
        :type RequestId: str
        """

        self.Result = None
        self.Description = None
        self.Confidence = None
        self.FakeFrames = None
        self.RequestID = None

    def _deserialize(self, params):
        self.Result = params.get("Result")
        self.Description = params.get("Description")
        self.Confidence = params.get("Confidence")
        # if video:
        #     self.FakeFrames = params.get("FakeFrames")
        self.RequestId = params.get("RequestId")

class DeepfakeClient(faceid_client.FaceidClient):

    def Deepfake(self, request):
        """使用动作活体检测模式前，需调用本接口获取动作顺序。

        :param request: Request instance for GetActionSequence.
        :type request: :class:`tencentcloud.faceid.v20180301.models.GetActionSequenceRequest`
        :rtype: :class:`tencentcloud.faceid.v20180301.models.GetActionSequenceResponse`

        """
        try:
            params = request._serialize()
            body = self.call("AntiFakesImage", params)
            response = json.loads(body)
            if "Error" not in response["Response"]:
                model = DeepfakeResponse()
                model._deserialize(response["Response"])
                return model
            else:
                code = response["Response"]["Error"]["Code"]
                message = response["Response"]["Error"]["Message"]
                reqid = response["Response"]["RequestId"]
                raise TencentCloudSDKException(code, message, reqid)
        except Exception as e:
            if isinstance(e, TencentCloudSDKException):
                raise
            else:
                raise TencentCloudSDKException(e.message, e.message)

class Sample:
    def __init__(self, style_code, fitness_score=-1):
        """
        value is a tensor
        """
        self.style_code = style_code
        self.fitness_score = fitness_score

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return base64.b64encode(fp.read()).decode()

def tencent(image_path):
    try:
        time.sleep(0.5)
        cred = credential.Credential("AKIDTs0sXwEd5HSM0SucYGvQZS6UlmW5NZR7", "kJgnlTu9Czd089thGagVXRn4C5h3Uemf")
        httpProfile = HttpProfile()
        httpProfile.endpoint = "faceid.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = DeepfakeClient(cred, "ap-shanghai", clientProfile)

        req = DeepfakeRequest()
        params = '{\"ImageBase64\":\"%s\"}' % (str(get_file_content(image_path)))

        req.from_json_string(params)

        resp = client.Deepfake(req)
        confidence = resp.Confidence

        return 1-confidence
    except TencentCloudSDKException as err:
        print(err)
        return 1.0


    except TencentCloudSDKException as err:
        print(err)

def baidu(image_path):
    APP_ID = '28026126'
    API_KEY = 'MSwOQ7fjBljYPMafsfwcIdmj'
    SECRET_KEY = 'u1IY1ajVqttEb447qL8vknrNQmVZO42r'

    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    try:
        options = {}
        options['face_field'] = 'spoofing'
        result = client.faceverify(get_file_content(image_path), options)
        result = result['result']
        spoofing = result['face_list'][0]['spoofing']
        spoofing = float(spoofing)
        return (spoofing / 0.00048) * 0.5
        # return spoofing
    except:
        return 1.0

def duckduckgoose(image_path):
    try:
        chromeOptions = webdriver.ChromeOptions()
        time.sleep(2)
        # try:
        # 设置代理
        chromeOptions.add_argument("--proxy-server=https://127.0.0.1:7890")
        chromeOptions.add_argument("--proxy-server=http://127.0.0.1:7890")
        browser = webdriver.Chrome(chrome_options=chromeOptions)
        browser.get('https://www.duckduckgoose.ai/demo')
        xx = browser.page_source
        element_input = browser.find_element("id", "fileElem")

        # 模拟文件上传
        element_input.send_keys(image_path)

        # 模拟点击
        browser.find_element("id", "analyzeButton").click()

        while (True):
            fake_score = browser.find_element("id", "fakeProb")
            if fake_score.text is not "":
                break
        score = float(fake_score.text[:-1])
        print("real-time score:", score)
        browser.close()
        return score/100.0
    except:
        return 1.0

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
    loss_disc += F.softplus(-fake_pred).mean()
    if first_dis:
        loss = 0
        loss_dis_init = loss_disc
        first_dis = False
    else:
        loss = loss_disc - loss_dis_init
    return loss * 1000

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
        loss = torch.tensor(0).cuda()
        loss_dis_init = loss_disc
        first_dis = False
    else:
        loss = loss_disc - loss_dis_init
        if loss < 0:
            loss = torch.tensor(0).cuda()
    return loss * 1000, loss_disc

def initpopluation(alpha, latent, nothing, image_name, output_path, configs_all_semantic, noise_in, first_dis, loss_dis_init, original_image, api, n = 500):
    # 1. get 500 style code
    channel_number = len(alpha)
    sample = []
    all_style_code = []
    weights = []
    for i in np.arange(-5.0, 5.0, 0.1):
        if np.abs(i) < 3:
            weights.append(0.6)
        elif np.abs(i) >= 3 and np.abs(i) < 4:
            weights.append(0.3)
        elif np.abs(i) >= 4:
            weights.append(0.1)
    for i in range(n):
        sample.append(random.choices(list(np.arange(-5.0, 5.0, 0.1)), k=channel_number, weights=weights))
    all_style_code.append(latent)
    for item in sample:
        tmp_latent = []
        for latent_code in latent:
            tmp_latent.append(latent_code.detach().clone())

        count = 0
        for config in configs_all_semantic:
            lindex = config[0]
            cindex = config[1]
            tmp_latent[lindex][0][0][cindex] = tmp_latent[lindex][0][0][cindex] + item[count]
            count += 1
        all_style_code.append(tmp_latent)

    # 2. get normal style code
    loss = []
    times = 0
    for style_code in all_style_code:
        img_gen, _ = g_ema([style_code], input_is_latent=True, randomize_noise=False, noise=noise_in,
                           input_is_stylespace=True)

        # store for api test
        img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
        img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
        img_save.save(os.path.join(output_path, str(times)+"_"+image_name))

        # dis loss
        if first_dis:
            loss_score, first__score = cal_dis_loss(style_code, first_dis, loss_dis_init)
            loss_dis_init = first__score
            first_dis = False
        else:
            loss_score, first__score = cal_dis_loss(style_code, first_dis, loss_dis_init)
        # id loss
        loss_id = cal_id_loss(img_tensor, original_image, original_image)

        loss.append(loss_id.item() + loss_score.item())
        times += 1
    # get n min loss style code
    popluation = []
    loss = np.array(loss)
    loss_index = np.argsort(loss)[:100]
    weights_v2 = []
    for i in loss_index:
        weights_v2.append(loss[i])
    popluation_index = random.choices(list(loss_index), k=50, weights=weights_v2)
    popluation_stylecode = []
    for i in popluation_index:
        popluation_stylecode.append(all_style_code[i])

    # 3. get fitness score
    fitness_score = []
    for i in popluation_index:
        image_path = os.path.join(output_path, str(i) + "_" + image_name)
        if api is "tencent":
            fitness_score.append(tencent(image_path))
        elif api is "baidu":
            fitness_score.append(baidu(image_path))
        elif api is "duckduckgoose":
            fitness_score.append(duckduckgoose(image_path))
    return popluation_stylecode, popluation_index, fitness_score

def find_elite(popluation_stylecode, fitness_score):
    fitness_score = np.array(fitness_score)
    fitness_score_sort_index = np.argsort(fitness_score)
    return Sample(popluation_stylecode[fitness_score_sort_index[0]], fitness_score[fitness_score_sort_index[0]])

def get_parents(k, fitness_score):
    weights = fitness_score
    parents_ind = random.choices(list(range(len(weights))), weights=weights, k=2*k)
    parents1_ind = parents_ind[:k]
    parents2_ind = parents_ind[k:]

    return parents1_ind, parents2_ind

def crossover(parents1_ind, parents2_ind, popluation_stylecode, fitness_score):
    fitness_score_torch = torch.from_numpy(np.array(fitness_score))
    parents1_fitness_scores = fitness_score_torch[parents1_ind]
    parents2_fitness_scores = fitness_score_torch[parents2_ind]
    p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores))
    parents1 = []
    parents2 = []
    for i in range(len(parents1_ind)):
        parents1.append(popluation_stylecode[parents1_ind[i]])
        parents2.append(popluation_stylecode[parents2_ind[i]])

    # select the channel to tensor
    children = []
    for i in range(len(parents1)):
        stylecode = []
        for j in range(len(parents1[i])):
            mask = torch.rand_like(parents1[i][j])
            mask = (mask < p[i]).float()
            stylecode.append(mask*parents1[i][j] + (1.-mask)*parents2[i][j])
        # xx = []
        # for z in range(len(stylecode)):
        #     xx.append((stylecode[z] - parents1[i][z]).detach().cpu().numpy())
        # yy = []
        # for z in range(len(stylecode)):
        #     yy.append((stylecode[z] - parents2[i][z]).detach().cpu().numpy())

        children.append(stylecode)

    return children

def mutate(children, configs_all_semantic, mutate_prob):
    for i in range(len(children)):
        for j in range(len(children[i])):
            mask = torch.rand_like(children[i][j])
            mask = torch.squeeze(mask)

            edit = (mask < mutate_prob).float()
            for k in range(len(edit)):
                if [j, k] in configs_all_semantic and edit[k] == 1:
                    children[i][j][0][0][k] = children[i][j][0][0][k] + mask[k]

    return children

def compute_fitness(mutated_children, noise_in, image_name, api):
    tmp_save_path = "/8T/work/search/dataset/study/StyleGAN/tmp"
    os.makedirs(tmp_save_path, exist_ok=True)
    times = 0
    fitness_score = []
    for stylecode in mutated_children:
        img_gen, _ = g_ema([stylecode], input_is_latent=True, randomize_noise=False, noise=noise_in,
                           input_is_stylespace=True)
        # store for api test
        img_tensor = transforms.Resize((299, 299))(((img_gen.detach().cpu() + 1) / 2).clamp_(0, 1))
        img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
        img_save.save(os.path.join(tmp_save_path, str(times) + "_" + image_name))

        image_path = os.path.join(tmp_save_path, str(times) + "_" + image_name)
        if api is "tencent":
            pred = tencent(image_path)
        elif api is "baidu":
            pred = baidu(image_path)
        elif api is "duckduckgoose":
            pred = duckduckgoose(image_path)
        fitness_score.append(pred)

    return fitness_score

def produce_next_generation(elite, popluation_stylecode, fitness_score, configs_all_semantic, noise_in, image_name, mutate_prob, api):
    parents1_ind, parents2_ind = get_parents(len(popluation_stylecode) - 1, fitness_score)
    children = crossover(parents1_ind, parents2_ind, popluation_stylecode, fitness_score)
    mutated_children = mutate(children, configs_all_semantic, mutate_prob)
    mutated_children.append(elite.style_code)
    fitness_score_new = compute_fitness(mutated_children, noise_in, image_name, api)

    return mutated_children, fitness_score_new



# Attack
def main(nothing, image_name, output_path, output_inversion_path, configs_all_semantic, latent_code_init, noise, target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init, result_path, mutate_prob, api):
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
                                      randomize_noise=False, noise=noise_in)
    latent = [s.detach().clone() for s in dlatents_loaded]  #S

    # 1. Define our stylespace channels that are edited
    alpha = []
    count = 0
    for config in configs_all_semantic:
        lindex = config[0]
        cindex = config[1]
        if lindex == 6 and cindex == 228 and (api is "tencent"):  # glasses
            alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy() - 25))
            latent[lindex][0][0][cindex] = latent[lindex][0][0][cindex] - 25
        else:
            alpha.append(torch.from_numpy(latent[lindex][0][0][cindex].detach().cpu().numpy()))
        count += 1

    # if score < 0.5 , directly exit
    tmp_save_path = "/8T/work/search/dataset/study/StyleGAN/tmp"
    os.makedirs(tmp_save_path, exist_ok=True)
    img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, noise=noise_in,
                       input_is_stylespace=True)
    # store for api test
    img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
    img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
    img_save.save(os.path.join(tmp_save_path, image_name))

    image_path = os.path.join(tmp_save_path, image_name)
    if api is "tencent":
        pred = tencent(image_path)
    elif api is "baidu":
        pred = baidu(image_path)
    elif api is "duckduckgoose":
        pred = duckduckgoose(image_path)

    if pred < 0.5:
        img_save.save(os.path.join(result_path, image_name))
        success = True
        return success

    popluation_stylecode, popluation_index, fitness_score = initpopluation(alpha, latent, nothing, image_name, output_path, configs_all_semantic, noise_in, first_dis, loss_dis_init, original_image, api)
    # Generation
    success = False
    fail_counts = 0
    pre_score = 1
    for gen in range(50):
        elite = find_elite(popluation_stylecode, fitness_score)
        # print(f'elite at {gen}-th generation: {elite.fitness_score}')
        print(gen, ": elite: ", elite.fitness_score)
        if elite.fitness_score < 0.5:
            img_gen, _ = g_ema([elite.style_code], input_is_latent=True, randomize_noise=False, noise=noise_in,
                               input_is_stylespace=True)
            # store for api test
            img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
            img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
            img_save.save(os.path.join(result_path, image_name))
            success = True
            return success

        if pre_score <= elite.fitness_score:
            fail_counts += 1
        else:
            fail_counts = 0
        pre_score = elite.fitness_score
        if fail_counts > 10:
            return success

        popluation_stylecode, fitness_score = produce_next_generation(elite, popluation_stylecode, fitness_score, configs_all_semantic, noise_in, image_name, mutate_prob, api)

    return success

if __name__ == '__main__':
    """
        **************************************                                  
        0x00 Define parameters
        **************************************
    """
    # dir_path = "/8T/work/search/dataset/original_1024_1000/test"
    dir_path = "/8T/xiangtao/new/dataset/inversion/299/ff++/images/with_noise_100"
    latent_path = "/8T/xiangtao/new/dataset/inversion/299/ff++/latents/with_noise_100"
    noise_path = "/8T/xiangtao/new/dataset/inversion/299/ff++/noise/with_noise_100"
    output_path = "/8T/xiangtao/new/dataset/results/ff++/tencent_tmp/"
    output_inversion_path = "/8T/xiangtao/new/dataset/results/ff++/"
    result_path = "/8T/xiangtao/new/dataset/results/ff++/tencent/"
    nothing = False  # without quality and optimization
    api = "tencent"

    target_model_name = "CNNDetection"
    lambda_dis = 0.001
    lambda_id = 0.001
    attribute_channel = [[21, 16], [21, 16], [21, 16]]
    attribute_edit = [0, 15.0, 30.0]
    if all:
        lr_list = [0.01]
    else:
        lr_list = [0.01]


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
        model.load('/8T/xiangtao/new/code/multi-semantic/weights/F3-Net/4_510_best.pkl')
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



    """
        **************************************                                  
        0x02 Attack
        **************************************
    """
    configs_all_attributes = {
        "all":[[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339],
                      [15, 102], [5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
                      [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
                      [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
                      [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130], [12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
                       [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191], [11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
                        [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
                        [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481], [6, 228]],
        "pale_skin": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [14, 339],
                      [15, 102]],
        "hairstyle": [[5, 92], [6, 323], [6, 394], [6, 500], [8, 128], [6, 487], [6, 322], [3, 259], [6, 285], [5, 414],
                      [9, 295], [3, 187], [3, 374], [5, 85], [5, 318], [6, 114], [6, 175], [6, 188], [8, 45], [6, 208],
                      [6, 216], [6, 341], [6, 497], [6, 504], [8, 175], [5, 111], [5, 397], [6, 137], [6, 204],
                      [6, 313], [6, 343], [6, 413], [15, 97], [9, 18], [9, 118], [9, 130]],
        "hair_color": [[12, 479], [12, 266], [15, 106], [14, 146], [12, 456], [12, 424], [12, 330], [12, 206], [3, 302],
                       [3, 486], [17, 249], [17, 92], [14, 4], [12, 287], [11, 286], [17, 19], [15, 191]],
        "lipstick": [[11, 6], [11, 73], [11, 116], [12, 5], [14, 110], [14, 490], [15, 37], [15, 45], [15, 235],
                     [17, 66], [17, 247], [18, 35]],
        "opend_mouth": [[11, 447], [11, 86], [8, 17], [6, 501], [6, 378], [6, 202], [6, 21], [9, 91], [9, 117],
                        [9, 232], [11, 204], [11, 279], [9, 452], [9, 26], [8, 456], [8, 389], [8, 191], [8, 118],
                        [8, 85], [18, 0], [6, 259], [11, 374], [12, 177], [12, 59], [11, 481]],
        "bushy_eyebrow": [[9, 440], [11, 290], [11, 433], [11, 364], [12, 100], [12, 102], [12, 242], [12, 278],
                          [12, 315], [12, 325], [12, 455]],
        "eyebrow_shape": [[14, 2], [8, 6], [8, 39], [8, 56], [8, 503], [9, 233], [9, 340], [9, 407], [11, 194],
                          [11, 312], [11, 320], [9, 30]],
        "earring": [[8, 81], [11, 15], [15, 47]],
        "eyeball_position": [[9, 409], [12, 43], [12, 149], [20, 93], [18, 33], [18, 17], [17, 163], [20, 92], [20, 74],
                             [9, 510], [11, 35]],
        "eye_size": [[8, 78], [9, 167], [11, 87], [9, 454], [12, 64], [12, 315]],
        "galsses": [[2, 175], [2, 97], [3, 120], [5, 325], [3, 288], [6, 228]]
        }
    for key, value in configs_all_attributes.items():
        # 1. create folders
        tmp_output_path = os.path.join(output_path, key)
        print("Attribute:", key)
        configs_all_semantic = value
        if not os.path.exists(tmp_output_path):
            os.makedirs(tmp_output_path)
        if not os.path.exists(output_inversion_path):
            os.makedirs(output_inversion_path)
        os.makedirs(result_path, exist_ok=True)

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
                fail_count = main(nothing, image, tmp_output_path, output_inversion_path, configs_all_semantic,
                                  latent_code, noise, target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init, api)
        else:
            fail_all_count = 0 # count the number of fail attack
            for image in os.listdir(latent_path):
                image_name = image.split(".")[-2] + ".png"

                # 1. If the sample has already been attack, skip it
                # if image_name in os.listdir(result_path):
                #     continue
                # if image_name not in os.listdir(dir_path):
                #     continue

                # 2. get y for id loss
                original_image = Image.open(os.path.join(dir_path, image_name))
                original_image = original_image.convert("RGB")
                original_image = transforms.ToTensor()(original_image)

                # 3. get latent code
                mutate_probs = [0.1, 0.2]
                for mutate_prob in mutate_probs:
                    print("You are workong :", image_name)
                    latent_code = np.load(os.path.join(latent_path, image))
                    latent_code = torch.from_numpy(latent_code)

                    # 4. get noise
                    noise = torch.load(os.path.join(noise_path, image.split(".")[-2] + ".pt"), map_location=torch.device('cpu'))

                    # 5. attack
                    first_dis = True
                    loss_dis_init = 0
                    success = main(nothing, image_name, tmp_output_path, output_inversion_path, configs_all_semantic,
                                      latent_code, noise,  target_model_name, original_image, lambda_dis, lambda_id, attribute_channel, attribute_edit, lr_list, first_dis, loss_dis_init, result_path, mutate_prob, api)
                    if success:
                        break

