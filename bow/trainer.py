import sys
import math
import logging

import torch
from tqdm import tqdm

from bow.helpers import reduce_dict

def train_val_one_epoch_object_detector(model, optimizer,  lr_scheduler, dataloader, device, epoch, phase, scaler=None):

    pbar = tqdm(dataloader)
    pbar.set_description(f"Phase {phase}")

    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            # backward + optimize only if in training phase
            if phase == 'train':
                if scaler is not None:
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

    return losses_reduced, optimizer.param_groups[0]["lr"]


def train_object_detector(model, learning_rate, dataloaders, device, num_epochs, kfold_idx, num_folds, logger):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # lr scheduler
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(dataloaders["train"]) - 1)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
        
    for epoch in range(num_epochs):
            print()
            logging.info('Fold {}: Epoch {}/{}'.format(kfold_idx + 1,  epoch + 1, num_epochs))
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                # running_loss, running_preds, running_targets = 
                loss, lr = train_val_one_epoch_object_detector(model, optimizer, lr_scheduler, dataloaders[phase], device, epoch, phase)
                # some_metrics = train_val_single_epoch(model,criterion, optimizer, scheduler, dataloaders[phase], device, phase)
                
                logging.info(
                    'Fold {}/{}: '
                    'Epoch {}/{}: '
                    'Phase {}: '
                    'Loss {}: '
                    'LR {}: '.format(kfold_idx + 1, num_folds,
                                        epoch + 1, num_epochs,
                                        phase,
                                        loss, 
                                        lr))

                logger.log({"epoch": epoch + 1,
                            "loss": loss,
                            "lr": lr})
                
                
    return model

def train_val_one_epoch_classifier(model, optimizer,  lr_scheduler, dataloader, device, epoch, phase, scaler=None):

    pbar = tqdm(dataloader)
    pbar.set_description(f"Phase {phase}")

    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            # backward + optimize only if in training phase
            if phase == 'train':
                if scaler is not None:
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

    return losses_reduced, optimizer.param_groups[0]["lr"]


def train_classifier(model, learning_rate, dataloaders, device, num_epochs, kfold_idx, num_folds, logger):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # lr scheduler
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(dataloaders["train"]) - 1)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
        
    for epoch in range(num_epochs):
            print()
            logging.info('Fold {}: Epoch {}/{}'.format(kfold_idx + 1,  epoch + 1, num_epochs))
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                # running_loss, running_preds, running_targets = 
                loss, lr = train_val_one_epoch_object_detector(model, optimizer, lr_scheduler, dataloaders[phase], device, epoch, phase)
                # some_metrics = train_val_single_epoch(model,criterion, optimizer, scheduler, dataloaders[phase], device, phase)
                
                logging.info(
                    'Fold {}/{}: '
                    'Epoch {}/{}: '
                    'Phase {}: '
                    'Loss {}: '
                    'LR {}: '.format(kfold_idx + 1, num_folds,
                                        epoch + 1, num_epochs,
                                        phase,
                                        loss, 
                                        lr))

                logger.log({"epoch": epoch + 1,
                            "loss": loss,
                            "lr": lr})
                
                
    return model