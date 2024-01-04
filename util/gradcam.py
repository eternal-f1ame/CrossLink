import argparse
from typing import Any
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image

from util.datasets import GaussianBlur

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def gradmap(model, images, num_crops=6):
    """
    Gradient Map based on GradCAM
    Args:
        model: the model to be used
        imgs: the input images (N, C, H, W)
        num_crops: the number of crops to be extracted from the image
    """
    imgs = images.permute(1, 2, 0).cpu().numpy()
    model.eval()
    model.cuda()
    model = model.cuda()
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)
    input_tensor = preprocess_image(imgs, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=False,
                        aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    mean_ = np.mean(grayscale_cam)*2
    binmask = (255*(grayscale_cam > mean_)).astype(np.uint8)
    # extract the BBOX around largest connected componentS from the mask in sorted order
    kernel = np.ones((27, 27), np.uint8)
    closing = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, kernel)
    # contours
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # parameters
    minarea = closing.shape[0]*closing.shape[1]*0.01
    maxarea = closing.shape[0]*closing.shape[1]*0.25
    # obtain the bounding box around each contour
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # transformation for the local small crops
    local_transfo = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        GaussianBlur(p=0.5),
        normalize,
    ])
    local_transfo_ = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 0.5), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        GaussianBlur(p=0.5),
        normalize,
    ])

    crops = []
    c = 0
    while c<num_crops:
        cnt = contours[c%len(contours)]
        if cv2.contourArea(cnt) > maxarea:
            cnt = cnt.reshape(-1, 2)
            hull = cv2.convexHull(cnt)
            x, y, w, h = cv2.boundingRect(hull)
            for _ in range(2):
                crop = local_transfo_(images[:,:,y:y+h, x:x+w])
                crops.append(crop)
            c+=1
        elif cv2.contourArea(cnt) > minarea and cv2.contourArea(cnt) < maxarea:
            x, y, w, h = cv2.boundingRect(cnt)
            crop = local_transfo(images[:,:,y:y+h, x:x+w])
            crops.append(crop)
        c+=1
    crops = crops[:num_crops]
    return crops, grayscale_cam

class GFB(object):
    def __init__(self, model):
        super(GFB, self).__init__()
        self.model = model
        self.bilinear = 32
        self.conv16 = nn.Conv2d(12, self.bilinear, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def get_feature_map(self, imgx):
        model = self.model
        x = model.patch_embed(imgx)
        b, n, c = x.shape
        cls_tokens = model.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += model.pos_embed
        x = model.pos_drop(x)

        feature_map = []
        for blk in model.blocks:
            x = blk(x)
            feature_map.append(x)
        feature_map = torch.stack(feature_map, dim=1)
        # Make the feature map of the same size of the input image
        feature_map = F.interpolate(feature_map, size=imgx.shape[-2:], mode='bilinear', align_corners=False)
        return feature_map
    
    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)
        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        img = att_max[:, None, :, :]
        img = img.repeat(1, 12, 1, 1)

        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat_max = nn.functional.normalize(bp_out_feat, dim=1)
        return bp_out_feat_max, att_max
    
    def __call__(self, imgx):
        featmap = self.get_feature_map(imgx)
        bp_out_feat_max, att_max = self.max_mask(featmap)
        return bp_out_feat_max, att_max
    