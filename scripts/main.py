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
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import  KFold

# from bow.model import Model
from bow.dataset import WadhwaniBollwormDataset
from bow.transform import BaselineTrainTransform
from bow.helpers import seed_everything, get_dir, reset_wandb_env, generate_random_string

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
parser.add_argument('-ts', '--test_size',
                    help='test size for cross validation', default=0.17, type=float)
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
                    help='path to data folder', default='data/source', type=str)
parser.add_argument('-dd', '--download_data',
                    help='should we download the data?', default=False, type=bool)
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


def train_val_single_epoch(model, criterion, optimizer, scheduler, dataloader, device, phase):
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_preds = []
    running_targets = []

    pbar = tqdm(dataloader)
    pbar.set_description(f"Phase {phase}")

    # Iterate over data.
    for image_ids, imgs, bboxes, targets in pbar:
        field_ids = field_ids.to(device)
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(imgs, bboxes)
            # preds = torch.argmax(F.softmax(outputs, 1), 1)
            # loss = criterion(outputs, targets)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
                scheduler.step()

        # metrics
        # running_loss += loss.item()
        # running_targets.extend(targets.cpu().detach().numpy())
        # running_preds.extend(preds.cpu().detach().numpy())

    return None # running_loss, running_preds, running_targets

def train(model, criterion, learning_rate, dataloaders, device, num_epochs, kfold_idx, num_folds, logger):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloaders['train']))
        
    for epoch in range(num_epochs):
            print()
            logging.info('Fold {}: Epoch {}/{}'.format(kfold_idx +
                         1,  epoch + 1, num_epochs))
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                # running_loss, running_preds, running_targets = 
                some_metrics = train_val_single_epoch(model,criterion, optimizer, scheduler, dataloaders[phase], device, phase)

                logging.info(
                    'Fold {}/{}: '
                    'Epoch {}/{}: '
                    'Phase {}: '.format(kfold_idx + 1, num_folds,
                                        epoch + 1, num_epochs,
                                        phase))

                logger.log({"epoch": epoch + 1})
                
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
        download=args.download_data,
        save=True,
        train=True,
        max_cache_length=args.max_cache_length,
        transform=BaselineTrainTransform())
    kfold =  KFold(n_splits=args.splits, test_size=args.test_size, random_state=args.seed)

    # arrays of model from cross validation of each snapshots
    models = []

    for kfold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset.field_ids, dataset.targets)):

        logging.info(
            f'Fold {kfold_idx + 1} / {args.spits}: {len(train_indices)} trains, {len(val_indices)} vals')

        logging.info(f'Fold {kfold_idx + 1} / {args.spits}: Loading dataset')

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
        logging.info(f'Fold {kfold_idx + 1} / {args.spits}: preparing model...')
        # model = nn.Something
        # model = model.to(device)

        # loss function
        logging.info(f'Fold {kfold_idx + 1} / {args.spits}: preparing loss function...')
        # criterion = nn.CrossEntropyLoss()
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

        models.append(mdoel)

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