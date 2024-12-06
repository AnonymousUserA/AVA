# -*- coding: utf-8 -*-
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
from matplotlib import pyplot as plt
from utils.utils import ensure_checkpoint_exists
from global_directions.manipulate import Manipulator
from torchvision import transforms
from GetCode_real import get_real_latent
from MapTS import GetFs, GetBoundary, GetDt
from torchviz import make_dot
from models.stylegan2.model import Generator
from util_scripts import get_dis_loss
from util_target_model import FFD
from util_target_model import CNNDetection
from util_target_model import GramNet
from util_target_model import F3_Net
from util_target_model import Xception
from util_target_model import Efficientnetb7

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
def main(real_image, image_name, output_path, output_inversion_path, configs_all_semantic, latent_code_init, model,
         loss_func, target_model_name):
    try:
        if real_image:
            # define working file
            image_name = image_path.split("/")[-1]
            print("You are working in:", image_name)

            # fine-tune image's latent code
            ## 1. define
            experiment_type = "ffhq_encode"
            os.chdir('/8T/work/search/Semantic/multi-semantic/global_directions/content/encoder4editing')
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
            img_gen, _ = g_ema([latent_code_init.cuda()], input_is_latent=True, randomize_noise=False,
                               input_is_stylespace=False)
            img_tensor = transforms.Resize((299, 299))(((img_gen + 1) / 2).clamp_(0, 1))
            img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
            img_save.save(os.path.join(output_inversion_path, image_name))
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
                if key in ["noises.noise_0", "noises.noise_1", "noises.noise_2", "noises.noise_3", "noises.noise_4", "noises.noise_5", "noises.noise_6", "noises.noise_7", "noises.noise_8", "noises.noise_9", "noises.noise_10", "noises.noise_11", "noises.noise_12", "noises.noise_13", "noises.noise_14", "noises.noise_15", "noises.noise_16"]:
                    tmp = value.clone().detach().cuda()
                    tmp.requires_grad = True
                    noise_in.append(tmp)
            optimizer = optim.Adam(latent_in, lr=0.01)


            for i in range(10):
                # update latent
                w_ff = torch.stack(latent_in).unsqueeze(0).squeeze(2)
                img_gen, _ = g_ema([w_ff], input_is_latent=True, randomize_noise=False, noise=None,
                                   input_is_stylespace=False)
                gen_im = denorm1(torch.squeeze(img_gen, 0))
                loss_sim = cal_loss.cal_loss(ori_im, gen_im, latent_code_init)
                loss_dis = get_dis_loss(w_ff)
                # loss = loss_sim + loss_dis
                print(loss_dis)
                loss = -loss_dis

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            w_ff = torch.stack(latent_in).unsqueeze(0)
            img_gen, _ = g_ema([w_ff], input_is_latent=True, randomize_noise=False,
                               input_is_stylespace=False)
            img_tensor = transforms.Resize((1024, 1024))(((img_gen + 1) / 2).clamp_(0, 1))
            img_save = transforms.ToPILImage()(img_tensor.squeeze(0))
            img_save.save(os.path.join(output_inversion_path, image_name))

    except TypeError:
        print("exception !!!!!!!!!!!!!")


if __name__ == '__main__':
    dir_path = "/8T/work/search/dataset/original_1024_1000/test/"
    output_path = "/8T/work/search/dataset/study/StyleGAN/test/"
    output_inversion_path = "/8T/work/search/dataset/study/test/"
    real_image = True
    target_model_name = "CNNDetection"

    # 0. Define configs
    configs_all_attributes = {
        "pale_skin": [[21, 16], [14, 57], [12, 235], [12, 44], [14, 79], [14, 194], [14, 5], [24, 12], [14, 339],
                      [15, 102]],
        # "lighting": [[24, 12]],
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

    # 1. Load Model
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
                        '/8T/work/search/Semantic/multi-semantic/global_directions/pretrained/xception-b5690688.pth')
        model.load('/8T/work/search/Semantic/F3Net-main/checkpoints/gan_both_api/4_510_best.pkl')
        model.model.eval()

        # 2. Define Loss
        loss_func = nn.BCEWithLogitsLoss().cuda()
        #################
    elif target_model_name is "CNNDetection":
        # 0. Import
        from target_model.CNNDetection.networks.resnet import resnet50

        # 1. Define and Load Model
        model = resnet50(num_classes=1)
        # state_dict = torch.load("/8T/work/detection/CNNDetection/weights/blur_jpg_prob0.1.pth", map_location='cpu')
        state_dict = torch.load("/8T/work/search/Semantic/CNNDetection-master/checkpoints/blur_jpg_prob0.1_latest/model_epoch_best.pth", map_location='cpu')
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
        model = torch.load('/8T/work/search/Semantic/multi-semantic/target_model/GramNet/weights/stylegan-ffhq.pth')
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
        MODEL_DIR = "/8T/work/search/Semantic/multi-semantic/target_model/FFD/pretrained model/xcp_tmp"
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
        model = torch.load("/8T/work/adv/DEEPSEC-master/weights/xception/full_raw.p")
        model.eval()
        model.cuda()

        # 2. Define Loss
        loss_func = nn.CrossEntropyLoss().cuda()
    elif target_model_name is "patch_forensics":
        # 0. Import
        import yaml
        from target_model.patch_forensics.models import create_model


        # 1. Define and Load Model
        class Opt(object):
            def __init__(self, d):
                for a, b in d.items():
                    if isinstance(b, (list, tuple)):
                        setattr(self, a, [Opt(x) if isinstance(x, dict) else x for x in b])
                    else:
                        setattr(self, a, Opt(b) if isinstance(b, dict) else b)


        # 1.1. Load config
        with open(
                "/8T/work/search/Semantic/detection/patch-forensics/checkpoints/gp1a-gan-winversion_seed0_xception_block5_constant_p10_randresizecrop/opt.yml",
                'r', encoding='utf-8') as f:
            config = f.read()
        d = yaml.load(config, Loader=yaml.FullLoader)
        opt = Opt(d)
        # 1.2 Load
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        # 2. Define Loss
        loss_func = nn.BCEWithLogitsLoss().cuda()

    # 3. Attack
    for key, value in configs_all_attributes.items():
        tmp_output_path = os.path.join(output_path, key)
        print("Attribute:", key)
        configs_all_semantic = value
        if not os.path.exists(tmp_output_path):
            os.makedirs(tmp_output_path)
        if not os.path.exists(output_inversion_path):
            os.makedirs(output_inversion_path)

        if real_image:
            image_list = os.listdir(output_inversion_path)
            for image in os.listdir(dir_path):
                if image in image_list:
                    continue
                image_path = os.path.join(dir_path, image)
                ## 3. get images's latent code
                latent_code = get_real_latent(image_path)
                fail_count = main(real_image, image, tmp_output_path, output_inversion_path, configs_all_semantic,
                                  latent_code, model, loss_func, target_model_name)
        else:
            count = 0
            fail_all_count = 0
            image_list = os.listdir(dir_path)
            for image in os.listdir(
                    "/8T/work/search/Semantic/multi-semantic/global_directions/finetuned_latent_code/stylegan"):
                if count >= 100:
                    break

                count += 1
                image_name = image.split(".")[-2] + ".png"
                if image_name not in image_list:
                    continue
                # if image_name not in ["FaceSwap_159_175_0002.png",  "Face2Face_097_033_0001.png"]:
                #     continue
                if image_name in os.listdir(tmp_output_path):
                    continue
                # print("You are workong :", image_name)
                latent_code = np.load(os.path.join(
                    "/8T/work/search/Semantic/multi-semantic/global_directions/finetuned_latent_code/stylegan", image))
                latent_code = torch.from_numpy(latent_code)
                fail_count = main(real_image, image_name, tmp_output_path, output_inversion_path, configs_all_semantic,
                                  latent_code, model, loss_func, target_model_name)
                fail_all_count += fail_count
            print("fail_count:", fail_all_count)