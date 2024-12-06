# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : attack.py
# @Project     : multi-semantic
# @CreateTime  : 2022/3/12 下午4:28:20
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import pickle
import copy
import os
import sys
from argparse import Namespace
sys.path.append("content/encoder4editing")
from global_directions.content.encoder4editing.models.psp import pSp
from global_directions.content.encoder4editing.utils.common import tensor2im
#from MapTS import GetFs,GetBoundary,GetDt

from global_directions.manipulate import Manipulator

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_alignment(image_path):
    import dlib
    from global_directions.content.encoder4editing.utils.alignment import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def get_real_latent(image_path):
    # define for e4e
    experiment_type = 'ffhq_encode'
    # experiment_type = 'test'
    os.chdir('/8T/xiangtao/new/code/multi-semantic/global_directions/content/encoder4editing')
    EXPERIMENT_ARGS = {
        "model_path": "e4e_ffhq_encode.pt"
    }
    EXPERIMENT_ARGS['transform'] = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    # print('Model successfully loaded!')

    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")

    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path)
    else:
        input_image = original_image
    if input_image is None:
        return None
    input_image.resize(resize_dims)

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    def display_alongside_source_image(result_image, source_image):
        res = np.concatenate([np.array(source_image.resize(resize_dims)),
                              np.array(result_image.resize(resize_dims))], axis=1)
        return Image.fromarray(res)

    def run_on_batch(inputs, net):
        images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        if experiment_type == 'cars_encode':
            images = images[:, :, 32:224, :]
        return images, latents

    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
    torch.save(latents, 'latents.pt')

    # Display inversion:
    # res = display_alongside_source_image(tensor2im(result_image), input_image)
    # plt.imshow(res)
    #
    # plt.savefig("result.jpg")
    return latents


if __name__ == '__main__':
    img_path = "/8T/work/search/age_adults/1.jpg"
    get_real_latent(img_path)