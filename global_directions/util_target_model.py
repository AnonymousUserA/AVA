# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : util_target_model.py
# @Project     : multi-semantic
# @CreateTime  : 2022/9/14 下午4:08:50
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************
# Common
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
# from imageio import imread
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import PIL

def FFD(image, model, loss_func):
    """
    Args:
        image: tensor[NxCxWxH]

    Returns:
        loss
        fake_score
    """
    # # Import
    # from target_model.FFD.network.xception import Model
    # from target_model.FFD.network.templates import get_templates
    # # Define
    # ## 0. loss
    # loss_func = nn.CrossEntropyLoss().cuda()
    # # loss_func = nn.BCEWithLogitsLoss().cuda()
    #
    # ## 1. model
    # BACKBONE = 'xcp'
    # MAPTYPE = 'tmp'
    # TEMPLATES = None
    # MODEL_DIR = "/8T/work/search/Semantic/multi-semantic/target_model/FFD/pretrained model/xcp_tmp"
    # if MAPTYPE in ['tmp', 'pca_tmp']:
    #     TEMPLATES = get_templates()
    # MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
    # MODEL.load(75, MODEL_DIR)
    # MODEL.model.eval()
    # MODEL.model.cuda()

    ## 2. transform
    transform = transforms.Compose([
          # transforms.ToPILImage(),
          # transforms.Resize((299, 299)),
          # transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    ## 3. torch
    torch.backends.deterministic = True
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    ## 4. detection
    image_transformed = transform(image)
    pred = model.model(image_transformed.cuda())  # return change
    fake_score = F.softmax(pred)[0][1]

    ## 5. calculate loss
    loss = loss_func(pred, torch.tensor([0]).cuda())
    # loss = loss_func(fake_score.unsqueeze(0), torch.tensor([0.0]).float().cuda())

    return loss, fake_score

def CNNDetection(image, model, loss_func):
    """
    Args:
        image: tensor[NxCxWxH]

    Returns:
        loss
        fake_score
    """
    # # 0. Import
    # from target_model.CNNDetection.networks.resnet import resnet50
    #
    # # 1. Define and Load Model
    # model = resnet50(num_classes=1)
    # state_dict = torch.load("/8T/work/detection/CNNDetection/weights/blur_jpg_prob0.1.pth", map_location='cpu')
    # model.load_state_dict(state_dict['model'])
    # model.eval()
    # model.cuda()
    #
    # # 2. Define Loss
    # loss_func = nn.BCEWithLogitsLoss().cuda()

    # 3. Define transform
    trans = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 4. Detection
    image_transformed = trans(image)
    # xx = model(image_transformed.cuda())
    fake_score = model(image_transformed.cuda())

    # 5. Calculate Loss
    loss = loss_func(fake_score.squeeze(1), torch.tensor([0.0]).cuda().float())

    return loss, fake_score.sigmoid()

def GramNet(image, model, loss_func):
    # # 0. Import
    # import resnet18_gram as resnet
    #
    # # 1. Define and Load model
    # # model = resnet.resnet18(pretrained=True)  # resnet18_gram.resnet18() #pretrained=True)
    # model = torch.load('/8T/work/search/Semantic/multi-semantic/target_model/GramNet/weights/stylegan-ffhq.pth')
    # model.eval()
    # model.cuda()
    #
    # # 2. Define Loss
    # # loss_func = nn.CrossEntropyLoss().cuda()
    # loss_func = nn.BCEWithLogitsLoss().cuda()
    # loss_func = nn.NLLLoss().cuda()
    # 3. Define transform
    # transform = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    # 4. Detection
    # original code directly use [0,255] as input
    # image = image * 255
    permute = [2, 1, 0]
    image = image[:, permute]
    # # original code directly use BGR as input, so we should change our tensor to BGR from RGB
    # image_edit = image.detach().clone().cpu().numpy()
    # ims = np.zeros((1, 3, 1024, 1024))
    # ims[0, 0, :, :] = image_edit[0, 2, :, :]
    # ims[0, 1, :, :] = image_edit[0, 1, :, :]
    # ims[0, 2, :, :] = image_edit[0, 0, :, :]
    # image.data = torch.from_numpy(ims.astype("uint8")).float().cuda().data


    output = model(image.float().cuda())
    fake_score = F.softmax(output)[0][0]

    # 5. Calculate Loss
    # loss = loss_func(output, torch.tensor([1]).cuda())
    loss = loss_func(fake_score.unsqueeze(0), torch.tensor([0.0]).float().cuda())

    return loss, fake_score

def F3_Net(image, model, loss_func):
    # # 0. Import
    # from target_model.F3net.trainer import Trainer
    # # 1. Define and Load Model
    # # config
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # pretrained_path = 'pretrained/xception-b5690688.pth'
    # gpu_ids = [*range(osenvs)]
    # target_model = Trainer(gpu_ids, "Both", '/8T/work/search/Semantic/multi-semantic/global_directions/pretrained/xception-b5690688.pth')
    # target_model.load('/8T/work/search/Semantic/F3Net-main/checkpoints/gan_both_v2/2_500_best.pkl')
    # target_model.model.eval()
    #
    # # 2. Define Loss
    # loss_func = nn.BCEWithLogitsLoss().cuda()

    # 3. Define transform
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Detection
    image_transformed = transform(image)
    fake_score = model.forward(image_transformed.cuda()).sigmoid()

    # 5. Calculate Loss
    loss = loss_func(fake_score.squeeze(1), torch.tensor([0.0]).cuda().float())

    return loss, fake_score

def Efficientnetb7(image):
    # 0. Import
    from target_model.others.deepfake_detector import dfdetector

    # 1. Define and Load Model
    model, img_size, normalization = dfdetector.prepare_method(method="efficientnetb7_dfdc", dataset=None, mode='test')
    model.eval()
    model.cuda()

    # 2. Define Loss
    loss_func = nn.BCEWithLogitsLoss().cuda()

    # 3. Define transform
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_tmp = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Detection
    image_transformed = transform(image)
    image_tmp = image.detach().clone()
    image_tmp = transform_tmp(image_tmp.squeeze())
    image_transformed.data = image_tmp.unsqueeze(0).cuda().data

    # 4. Detection
    fake_score = model(image_transformed.cuda()).sigmoid()

    # 5. Calculate Loss
    loss = loss_func(fake_score.squeeze(0), torch.tensor([0.0]).cuda().float())

    return loss, fake_score

def Xception(image, model, loss_func):
    # 0. Define transform
    transform = transforms.Compose([
        # transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        # transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # 1. Detection
    image_transformed = transform(image)
    pred = model.model(image_transformed.cuda())  # return change
    fake_score = F.softmax(pred)[0][1]

    ## 2. calculate loss
    loss = loss_func(pred, torch.tensor([0]).cuda())
    # loss = loss_func(fake_score.unsqueeze(0), torch.tensor([0.0]).float().cuda())

    return loss, fake_score

def patch_forensics(image, model, loss_func):
    # 0. Define transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. Detection
    img_transformed = transform(image)
    # model.reset()
    inputs = dict(ims=img_transformed.cuda(), labels=torch.tensor([0]).long().cuda())
    model.set_input(inputs)
    model.test(True)
    predictions = model.get_predictions()
    fake_score = predictions[0][0]

    # 2. Calculate Loss
    loss = loss_func(predictions, torch.tensor([1]).cuda())
    # loss = loss_func(fake_score.unsqueeze(0), torch.tensor([0.0]).float().cuda())
    # print(loss)

    return loss, fake_score

def ResNet(image, model, loss_func):
    transform = transforms.Compose([
        # transforms.Resize((299, 299)),
        # transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # 1. Detection
    image_transformed = transform(image)
    pred = model(image_transformed.cuda())  # return change
    fake_score = F.softmax(pred)[0][1]

    ## 2. calculate loss
    # loss = loss_func(pred, torch.tensor([0]).cuda())
    loss = loss_func(fake_score.unsqueeze(0), torch.tensor([0.0]).float().cuda())

    return loss, fake_score




