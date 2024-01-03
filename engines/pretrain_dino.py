import sys
import math
import torch
from util import misc
from util import lars
from util import lr_sched
from typing import Iterable
from util.gradcam import gradmap


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None):
    student = model.student
    teacher = model.teacher
    teacher_without_ddp = model.teacher_without_ddp
    dino_loss = model.loss
    lr_schedule = lr_sched.cosine_scheduler(
        args.lr * (args.batch_size * misc.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = lr_sched.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = lr_sched.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    print_freq = 20
    accum_iter = args.accum_iter

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for data_iter_step, val in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.dual:
            images, images_, _ = val
        else:
            images, _ = val
            images_ = None

        # update weight decay and learning rate according to their schedule
        data_iter_step = len(data_loader) * epoch + data_iter_step  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[data_iter_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[data_iter_step]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        if args.dual:
            images_ = [im.cuda(non_blocking=True) for im in images_]
        else:
            images_ = images

        if args.gradcam:
            images[2:] = gradmap(student, images, args.local_crops_number - 2)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images_[:2]) # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = misc.clip_gradients(student, args.clip_grad)
            misc.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = misc.clip_gradients(student, args.clip_grad)
            misc.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[data_iter_step]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}