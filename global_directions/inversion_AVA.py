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
from torchviz import make_dot
from models.stylegan2.model import Generator
from util_target_model import FFD
from util_target_model import CNNDetection
from util_target_model import GramNet
from util_target_model import F3_Net
from util_target_model import Efficientnetb7
from util_target_model import patch_forensics
from util_target_model import Xception
from tqdm import tqdm
from collections import OrderedDict
from util_scripts import get_dis_loss
# define Methods
g_ema = Generator(1024, 512, 8)
weights = torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"]
g_ema.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.cuda()
# mean_latent = g_ema.mean_latent(4096)
# M = Manipulator(dataset_name='ffhq')


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
# def main(image_path, output_path, output_inversion_path, configs_all_semantic):
def main(real_image, image_path, output_latent_path, output_inversion_path, latent_code_init, output_noise_path):
    try:
        if real_image:
            # define working file
            image_name = image_path.split("/")[-1]
            print("You are working in:", image_name)

            # fine-tune image's latent code
            ## 1. define
            experiment_type = "ffhq_encode"
            os.chdir('/8T/xiangtao/new/code/multi-semantic/global_directions/content/encoder4editing')
            EXPERIMENT_ARGS = {}
            EXPERIMENT_ARGS['transform'] = transforms.Compose([transforms.ToTensor()])

            ## 2. get original image
            original_image = Image.open(image_path)
            original_image = original_image.convert("RGB")
            if experiment_type == "ffhq_encode":
                input_image = run_alignment(image_path)
            else:
                input_image = original_image
            img_transforms = EXPERIMENT_ARGS['transform']
            transformed_image = img_transforms(input_image)
            ori_im = transformed_image.cuda()

            ## 3. Define adam and grad
            ### 1. latent code
            latent_code_init = latent_code_init.squeeze(0).cuda()
            latent = latent_code_init.split(1, dim=0)
            latent_in = []
            for item in latent:
                tmp = item.clone().detach().cuda()
                tmp.requires_grad = True
                latent_in.append(tmp)

            ### 2. noise
            noise_in = []
            for key, value in weights.items():
                if key in ["noises.noise_0", "noises.noise_1", "noises.noise_2", "noises.noise_3", "noises.noise_4",
                           "noises.noise_5", "noises.noise_6", "noises.noise_7", "noises.noise_8", "noises.noise_9",
                           "noises.noise_10", "noises.noise_11", "noises.noise_12", "noises.noise_13",
                           "noises.noise_14", "noises.noise_15", "noises.noise_16"]:
                    tmp = value.clone().detach().cuda()
                    tmp.requires_grad = True
                    noise_in.append(tmp)
            optimizer = optim.Adam([{'params': latent_in, 'lr': 0.01,},
                                    {'params': noise_in, "lr": 0.01}])

            for i in tqdm(range(0), desc='Processing'):
                # update latent
                w_ff = torch.stack(latent_in).unsqueeze(0).squeeze(2)
                img_gen, _ = g_ema([w_ff], input_is_latent=True, randomize_noise=False, noise=noise_in,
                                   input_is_stylespace=False)
                gen_im = denorm1(torch.squeeze(img_gen, 0))
                loss_sim = cal_loss.cal_loss(ori_im, gen_im, latent_code_init)
                loss_dis = get_dis_loss(w_ff)
                loss = loss_sim + 0.0001 * loss_dis
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            w_ff = torch.stack(latent_in).unsqueeze(0).squeeze(2)
            np.save(os.path.join(output_latent_path, image_name.split(".")[-2] + ".npy"), w_ff.detach().cpu().numpy())
            noise_save = OrderedDict()
            name_list = ["noises.noise_0", "noises.noise_1", "noises.noise_2", "noises.noise_3", "noises.noise_4",
                           "noises.noise_5", "noises.noise_6", "noises.noise_7", "noises.noise_8", "noises.noise_9",
                           "noises.noise_10", "noises.noise_11", "noises.noise_12", "noises.noise_13",
                           "noises.noise_14", "noises.noise_15", "noises.noise_16"]
            for i in range(len(noise_in)):
                noise_save[name_list[i]] = noise_in[i]
            torch.save(noise_save, os.path.join(output_noise_path, image_name.split(".")[-2] + ".pt"))
            # img_gen, _ = g_ema([latent_code_init.unsqueeze(0)], input_is_latent=True, randomize_noise=False, noise=noise_in,
            #                    input_is_stylespace=False)
            img_gen, _ = g_ema([w_ff], input_is_latent=True, randomize_noise=False,
                               noise=noise_in,
                               input_is_stylespace=False)
            img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
            img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
            img_save.save(os.path.join(output_inversion_path, image_name))

    except TypeError:
        print("exception !!!!!!!!!!!!!")

if __name__ == '__main__':
    dir_path = "/8T/xiangtao/new/dataset/original_1024_100/test"
    output_latent_path = "/8T/xiangtao/new/dataset/inversion/stylegan_v2/latents/with_noise_100" #100 is the number of iteration
    output_noise_path = "/8T/xiangtao/new/dataset/inversion/stylegan_v2/noise/with_noise_100"
    output_inversion_path = "/8T/xiangtao/new/dataset/inversion/stylegan_v2/images/with_noise_100"
    real_image = True


    # create dir
    if not os.path.exists(output_latent_path):
        os.makedirs(output_latent_path)
    if not os.path.exists(output_inversion_path):
        os.makedirs(output_inversion_path)
    if not os.path.exists(output_noise_path):
        os.makedirs(output_noise_path)

    if real_image:
        for image in os.listdir(dir_path):
            # if image not in ["004426.png"]:
            #     continue
            if image in os.listdir(output_inversion_path):
                continue
            image_path = os.path.join(dir_path, image)
            ## 3. get images's latent code
            latent_code = get_real_latent(image_path)
            main(real_image, image_path, output_latent_path, output_inversion_path,latent_code, output_noise_path)