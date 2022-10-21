import os
import copy
import time
import pickle
import logging
import argparse
import datetime
from argparse import Namespace

import yaml
import torch
import wandb
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import  KFold
from torch.utils.data import Subset, DataLoader

from bow.model import InsectDetector
from bow.dataset import WadhwaniBollwormDataset
from bow.transform import BaselineTrainTransform
from bow.helpers import seed_everything, get_dir, reset_wandb_env, generate_random_string, reduce_dict

parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o', '--output_dir',
                    help='save path for trained models', default='results', type=str)

# experiment
parser.add_argument(
    '-n', '--name', help='name of experiment', default="agrifield-challenge", type=str)
parser.add_argument(
    '-off', '--offline', help='should we run the experiments offline?', default=True, type=bool)
parser.add_argument(
    '-s', '--seed', help='seed for experiments', default=42, type=int)
parser.add_argument('-dv', '--device',
                    help='cuda device to use', default=0, type=int)
parser.add_argument(
    '-ks', '--splits', help='number of splits for cross validation', default=10, type=int)
parser.add_argument(
    '-p', '--predict', help='predict the classes for the test data in a submission file', default=True, type=bool)
parser.add_argument('-ssp', '--sample_submission_path', help='path to the sample submssion path',
                    default='data/source/SampleSubmission.csv', type=str)

# data
parser.add_argument('-d', '--data_dir',
                    help='path to data folder', default='data/', type=str)
parser.add_argument('-b', '--batch_size',
                    help='batch size', default=256, type=int)
parser.add_argument('-w', '--num_workers',
                    help='number of workers for dataloader', default=8, type=int)
parser.add_argument('-cs', '--crop_size',
                    help='size of the crop image after transform', default=32, type=int)
parser.add_argument('-ml', '--max_cache_length',
                    help='max length of the cache of our data', default=512, type=int)

# model optimization & training
parser.add_argument('-ep', '--epochs',
                    help='number of training epochs', default=10, type=int)
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate', default=0.1, type=float)
parser.add_argument(
    '-sp', '--sweep_path', help='path to sweep configuration if we wish to start a sweep', default=None, type=str)

parser.add_argument(
    '-sc', '--sweep_count', help="number of runs to makefor the sweep", default=None, type=int)

initial_args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')


def train_val_one_epoch(model, optimizer,  lr_scheduler, dataloader, device, epoch, phase, scaler=None):
    
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


def train(model, criterion, learning_rate, dataloaders, device, num_epochs, kfold_idx, num_folds, logger):
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
                loss, lr = train_val_one_epoch(model, optimizer, lr_scheduler, dataloaders[phase], device, epoch, phase)
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
                
                
def main():
    sweep_run_name = f"{datetime.datetime.now().strftime(f'%H-%M-%ST%d-%m-%Y')}_{generate_random_string(5)}"

    # directory to save models and parameters
    results_dir = get_dir(f'{initial_args.output_dir}/{sweep_run_name}')

    # combine wwandb config with args to form old args (sweep)
    # dumb init to get configs
    wandb.init(dir=get_dir(initial_args.output_dir))
    args = Namespace(**(vars(initial_args) | dict(wandb.config)))
    wandb.join()

    # save hyperparameters
    with open(f'{results_dir}/hparams.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    logging.info(f'Preparing dataset...')

    seed_everything(args.seed)
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = WadhwaniBollwormDataset(
        args.data_dir,
        save=True,
        train=True,
        max_cache_length=args.max_cache_length,
        transform=BaselineTrainTransform(train=True))
    kfold =  KFold(n_splits=args.splits, random_state=args.seed, shuffle=True)

    # arrays of model from cross validation of each snapshots
    models = []

    for kfold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset.bboxes)):

        logging.info(
            f'Fold {kfold_idx + 1} / {args.splits}: {len(train_indices)} trains, {len(val_indices)} vals')

        logging.info(f'Fold {kfold_idx + 1} / {args.splits}: Loading dataset')

        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

        dataloaders = {
            "train": DataLoader(train_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers),
            "val": DataLoader(val_ds,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
        }

        # model
        logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing model...')
        model = InsectDetector(num_classes=len(dataset.bollworms))
        model = model.to(device)

        # loss function
        logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing loss function...')
        criterion = None
        # criterion.to(device)

        # reset wandb env
        reset_wandb_env()

        # wandb configs
        run = wandb.init(project=args.name,
                         name=f'kfold_{kfold_idx + 1}',
                         group=sweep_run_name,
                         dir=get_dir(args.output_dir),
                         config=args,
                         reinit=True)

        # get a snapshot of model for this k fold
        logging.info(f'Fold {kfold_idx + 1} / {args.splits}: getting model snapshots...')
        model = train(model, criterion, args.learning_rate, dataloaders, device, num_epochs=args.epochs, kfold_idx=kfold_idx, num_folds=args.splits, logger=run)

        models.append(model)

        wandb.join()

    sweep_run = wandb.init(project=f"{args.name}-sweeps",
                           name=sweep_run_name,
                           config=args,
                           dir=get_dir(args.output_dir))

    sweep_run.log({})
    wandb.join()

    # save models
    for i, m in enumerate(models):
        torch.save(m.state_dict(), f"{results_dir}/model_{i}.pth")
        
if __name__ == "__main__":
    if initial_args.sweep_path:

        import yaml
        with open(initial_args.sweep_path, "r") as stream:
            try:
                sweep_configuration = yaml.safe_load(stream)
                sweep_id = wandb.sweep(
                    sweep=sweep_configuration, project=initial_args.name)
                wandb.agent(sweep_id, function=main,
                            project=initial_args.name, count=initial_args.sweep_count)

            except yaml.YAMLError as exc:
                logging.error(
                    f"Couldn't load the sweep file. Make sure {initial_args.sweep_path} is a valid path")
                logging.warn("doing a normal run")
                main()
    else:
        main()