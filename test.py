import math
import torch
import torchvision
from net import  Net
from data import load_images_names_in_data_set, get_bb_of_gt_from_pascal_xml_annotation
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw, ImageFont
from utils import cal_iou, reward_func, reward_func_gfirs
import cv2
import time
from train import DQN
from repvgg_in import deploy_model


start = time.time()

# Hyperparameters
NUM_ACTIONS = 9
his_actions = 10
subscale = 0.2
NUM_STATES = 5*5*1280 + his_actions * NUM_ACTIONS
path_voc = "home/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/"


def init_process(image, transform=None):
    if transform:
        image = transform(image)
    return image.unsqueeze(0)


def inter_process(image, bbx, transform=None):
    (left, upper, right, lower) = (bbx[0], bbx[2], bbx[1], bbx[3])
    image_crop = image.crop((left, upper, right, lower))
    if transform:
        image_crop = transform(image_crop)
    return image_crop.unsqueeze(0)


def update_bbx(bbx, action):
    """Update bounding box based on action"""
    new_bbx = np.zeros(4)
    if action == 0:
        new_bbx[0] = bbx[0] - (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 1:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1] + (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 2:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] - (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale
    elif action == 3:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3] + (bbx[3] - bbx[2]) * subscale
    elif action == 4:
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 5:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[3]
    elif action == 6:
        new_bbx[0] = bbx[0] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2] + (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] - (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 7:
        new_bbx[0] = bbx[0] - (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[1] = bbx[1] + (bbx[1] - bbx[0]) * subscale * 1 / 2
        new_bbx[2] = bbx[2] - (bbx[3] - bbx[2]) * subscale * 1 / 2
        new_bbx[3] = bbx[3] + (bbx[3] - bbx[2]) * subscale * 1 / 2
    elif action == 8:
        new_bbx = bbx
    return new_bbx

def main(args):
    device = torch.device("cuda:1" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    image_names = np.array(load_images_names_in_data_set('aeroplane_test', path_voc))

    deploy_model.gap = nn.Sequential(nn.AdaptiveAvgPool2d(5))
    deploy_model.linear = nn.Sequential()
    feature_exactrator = deploy_model.to(device)

    single_plane_image_names = []
    single_plane_image_gts = []

    dqn = DQN(device)
    EPISILO = args.EPISILO
    dqn.load()

    miou = 0
    iou_sum = 0
    precision50 = 0


    for image_name in image_names:
        annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
        if len(annotation) > 1:
            continue
        single_plane_image_names.append(image_name)
        single_plane_image_gts.append(annotation[0][1:])

    trans = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    ep_reward = 0
    n = 0
    for index, image_name in enumerate(single_plane_image_names):
        image_path = os.path.join(path_voc + "JPEGImages", image_name + ".jpg")
        image_original = Image.open(image_path)
        width, height = image_original.size
        bbx_gt = single_plane_image_gts[index]

        image = init_process(image_original, trans).to(device)

        bbx = [0, width, 0, height]
        history_action = np.zeros(his_actions * NUM_ACTIONS)
        with torch.no_grad():
            vector = feature_exactrator(image).cpu().detach().numpy().reshape(5*5*1280)
        state = np.concatenate([history_action, vector])
        step = 0
        while step < 50:
            action = dqn.choose_action(state, EPISILO)
            new_bbx = update_bbx(bbx, action)

            action_vec = np.zeros(NUM_ACTIONS)
            action_vec[action] = 1.0
            history_action = np.concatenate([history_action[NUM_ACTIONS:], action_vec])

            with torch.no_grad():
                vector = feature_exactrator(
                    inter_process(image_original, new_bbx, trans).to(device)).cpu().detach().numpy().reshape(
                    5*5*1280)
            next_state = np.concatenate([history_action, vector])

            if action == 8 or step == 49:
                draw = ImageDraw.Draw(image_original)
                draw.rectangle([bbx_gt[0], bbx_gt[2], bbx_gt[1], bbx_gt[3]], outline='green', width=2)
                draw.rectangle([bbx[0], bbx[2], bbx[1], bbx[3]], outline='red', width=2)
                font = ImageFont.truetype('home/fonts/Droid-Sans.ttf', 30)
                iou = cal_iou(bbx, bbx_gt)
                draw.text([bbx[0], bbx[2]], str(round(iou, 2)), fill=(255, 0, 0), font=font)

                iou_sum += iou
                if iou > 0.5:
                    precision50 += 1
                image_original.save('home/visualization/visualization_aeroplane_test' + "/" + image_name + ".jpg")
                n += 1
                break

            state = next_state
            bbx = new_bbx
            step += 1

    miou = iou_sum/n
    p50 = precision50/n
    print("miou = {}, precision50 = {}".format(miou, p50))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Object Detection with Deep Reinforcement Learning')
    parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_gpu', default=True, action='store_true')
    parser.add_argument('--EPISILO', type=int, default=0)

    main(parser.parse_args())

end = time.time()
print('Running time: %s Seconds' % (end - start))