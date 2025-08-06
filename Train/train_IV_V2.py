# -*- coding: utf-8 -*-
"""
Main training script for YOLOv5-Lite-IV model.

This script handles the training process for the dual-stream model for
infrared and visible light object detection. It loads the custom model,
prepares the dataset, and runs the training loop.

Usage:
    $ python train.py --data data/your_dataset.yaml --cfg models/yolov5l_IV_V2.yaml --weights '' --batch-size 8
"""
import argparse
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

# Add project root to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Assuming `train.py` is in the project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- Local Imports ---
# These imports assume the standard YOLOv5 `models` and `utils` directory structure.
# Ensure these directories are present in your repository.
from models.yolo_IV_V2 import Model
from utils.datasets_IV import create_dataloader  # Custom dataloader for dual inputs
from utils.loss_IV import ComputeLoss
from utils.general import (labels_to_class_weights, increment_path, init_seeds,
                           strip_optimizer, check_dataset, check_img_size,
                           check_yaml, check_suffix, print_args,
                           set_logging, one_cycle, colorstr)
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, intersect_dicts
from utils.metrics import fitness

# It's better to have a validation script, but for simplicity, we can omit it for the initial repo.
# import val as validate

LOGGER = logging.getLogger(__name__)


def train(hyp, opt, device):
    """
    Main training function.
    hyp: Hyperparameters dictionary.
    opt: Command-line options.
    device: PyTorch device.
    """
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers

    # --- Directories ---
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # --- Hyperparameters & Settings ---
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # --- System Setup ---
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(1)
    data_dict = check_dataset(data)
    train_path_visible, train_path_infrared = data_dict['train_visible'], data_dict['train_infrared']
    # val_path_visible, val_path_infrared = data_dict['val_visible'], data_dict['val_infrared'] # Uncomment if you have a validation set
    nc = 1 if single_cls else int(data_dict['nc'])
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} in {data}'

    # --- Model ---
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)
        model = Model(cfg or ckpt['model'].yaml, ch_visible=3, ch_infrared=1, nc=nc, anchors=hyp.get('anchors')).to(
            device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = Model(cfg, ch_visible=3, ch_infrared=1, nc=nc, anchors=hyp.get('anchors')).to(device)

    # --- Optimizer ---
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # --- Scheduler ---
    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # --- EMA, Resume, etc. ---
    ema = ModelEMA(model)
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']
        del ckpt, csd

    # --- Dataloaders ---
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    train_loader, dataset = create_dataloader([train_path_visible, train_path_infrared], imgsz, batch_size, gs,
                                              single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=-1,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '))
    nb = len(train_loader)

    # --- Loss Function ---
    compute_loss = ComputeLoss(model)

    # --- Start Training ---
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    scaler = amp.GradScaler(enabled=cuda)

    LOGGER.info(f'Image sizes {imgsz} train\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'shared', 'unique'))
        pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()

        for i, (imgs_visible, imgs_infrared, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs_visible = imgs_visible.to(device, non_blocking=True).float() / 255.0
            imgs_infrared = imgs_infrared.to(device, non_blocking=True).float() / 255.0

            # --- Warmup ---
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # --- Forward ---
            with amp.autocast(enabled=cuda):
                outputs, loss_shared_list, loss_unique_list = model(imgs_infrared, imgs_visible)

                # Calculate detection loss
                det_loss, loss_items = compute_loss(outputs, targets.to(device))

                # Calculate custom fusion losses
                shared_loss = sum(loss_shared_list) if loss_shared_list else 0.0
                unique_loss = sum(loss_unique_list) if loss_unique_list else 0.0

                # Combine losses with weights
                total_loss = (opt.det_loss_w * det_loss +
                              opt.shared_loss_w * shared_loss +
                              opt.unique_loss_w * unique_loss)

            # --- Backward ---
            scaler.scale(total_loss).backward()

            # --- Optimize ---
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # --- Log ---
            loss_all = torch.cat((loss_items, torch.tensor([shared_loss, unique_loss], device=device)))
            mloss = (mloss * i + loss_all[:3]) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, loss_all[3], loss_all[4]))

        # --- End of Epoch ---
        scheduler.step()

        # Save model
        if (not nosave) or ((epoch + 1) == epochs):
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,  # Note: validation logic removed for simplicity
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict()}

            # Save last, best and periodic checkpoints
            torch.save(ckpt, last)
            # if best_fitness == fi: # Validation logic needed to determine best
            #     torch.save(ckpt, best)
            if (opt.save_period > 0) and (epoch % opt.save_period == 0):
                torch.save(ckpt, w / f'epoch{epoch}.pt')
            del ckpt

    # --- End of Training ---
    LOGGER.info(f'\n{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    # --- Core arguments ---
    parser.add_argument('--weights', type=str, default='', help='initial weights path (optional)')
    parser.add_argument('--cfg', type=str, required=True, help='model.yaml path')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    # --- Loss weights ---
    parser.add_argument('--det-loss-w', type=float, default=1.0, help='weight for detection loss')
    parser.add_argument('--shared-loss-w', type=float, default=0.5, help='weight for shared feature loss')
    parser.add_argument('--unique-loss-w', type=float, default=0.5, help='weight for unique feature loss')

    # --- Other training options ---
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    set_logging()

    # Sanity checks
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_yaml(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    # Set save directory
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Select device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Start training
    train(opt.hyp, opt, device)