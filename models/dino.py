# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial, reduce
from operator import mul

import numpy as np
from PIL import Image

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from util import datasets
from util.crop import MultiCropWrapper
from util.misc import trunc_normal_, has_batchnorms
from util.crop import MultiCropWrapper
from models import vision_transformer as vits

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
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

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            datasets.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            datasets.GaussianBlur(0.1),
            datasets.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            datasets.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class DualDataAugmentationDINO(DataAugmentationDINO):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        super().__init__(global_crops_scale, local_crops_scale, local_crops_number)
    def __call__(self, image, image_):
        crops = [], crops_ = []
        crops.append(self.global_transfo1(image)), crops_.append(self.global_transfo1(image_))
        crops.append(self.global_transfo2(image)), crops_.append(self.global_transfo2(image_))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image)), crops_.append(self.local_transfo(image_))
        return crops, crops_

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOViT(nn.Module):
    def __init__(self, model_name, local_crops_number, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp_epochs, epochs, args):
        super().__init__()

        # ============ preparing student and teacher ... ============
        student = vits.__dict__[model_name](
                drop_path_rate=args.drop_path_rate,  # stochastic depth
            )
        teacher = vits.__dict__[model_name]()
        embed_dim = student.embed_dim
        student = MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        teacher = MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )
        # move networks to gpu
        student, teacher = student.cuda(), teacher.cuda()
        # synchronize batch norms (if any)
        print(has_batchnorms(student))
        if has_batchnorms(student):
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            # we need DDP wrapper to have synchro batch norms working...
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            print("Using Synchronized BatchNorm:", args.gpu)
            teacher_without_ddp = teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            teacher_without_ddp = teacher
        student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False

        self.student = student
        self.teacher = teacher
        self.teacher_without_ddp = teacher_without_ddp
        self.local_crops_number = local_crops_number
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.epochs = epochs

        # ============ preparing loss ... ============
        self.loss = DINOLoss(
            args.out_dim,
            local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            epochs,
        ).cuda()

    def forward(self, images):
        student_output = self.student(images)
        with torch.no_grad():
            teacher_output = self.teacher(images[:2])
        return student_output, teacher_output
    
    def train_update(self, epochs, iter_per_epoch, cur_epoch):
        self.iter_per_epoch = iter_per_epoch
        self.max_iter = epochs * iter_per_epoch
        self.epoch = cur_epoch
        self.iteration = cur_epoch * iter_per_epoch
