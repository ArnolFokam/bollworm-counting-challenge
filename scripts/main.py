import logging
import argparse
import datetime
from argparse import Namespace

import yaml
import torch
import wandb
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

from bow.model import ImageClassifier, InsectDetector
from bow.dataset import WadhwaniBollwormDataset
from bow.trainer import train_classifier, train_object_detector
from bow.transform import BaselineTrainTransform
from bow.helpers import seed_everything, get_dir, reset_wandb_env, generate_random_string, reduce_dict

parser = argparse.ArgumentParser(description='Ensemble training script')

# general
parser.add_argument('-o', '--output_dir',
                    help='save path for trained models', default='results', type=str)

# experiment
parser.add_argument(
    '-n', '--name', help='name of experiment', default="bollworm-challenge", type=str)
parser.add_argument(
    '-doj', '--do_object_detection', help='train object detection models?', default=True, type=bool)
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
parser.add_argument('-ml', '--max_cache_length',
                    help='max length of the cache of our data', default=512, type=int)
parser.add_argument('-imw', '--image_width',
                    help='width of an image', default=256, type=int)
parser.add_argument('-imh', '--image_height',
                    help='height of an image', default=256, type=int)

# model optimization & training
parser.add_argument('-ep', '--epochs',
                    help='number of training epochs', default=10, type=int)
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate', default=0.0005, type=float)
parser.add_argument(
    '-sp', '--sweep_path', help='path to sweep configuration if we wish to start a sweep', default=None, type=str)

parser.add_argument(
    '-sc', '--sweep_count', help="number of runs to makefor the sweep", default=None, type=int)

initial_args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')

                
                
def main():
    if initial_args.do_object_detection:
        sweep_run_name = f"{datetime.datetime.now().strftime(f'%H-%M-%ST%d-%m-%Y')}_{generate_random_string(5)}"

        # directory to save models and parameters
        results_dir = get_dir(f'{initial_args.output_dir}/od/{sweep_run_name}')

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = WadhwaniBollwormDataset(
            args.data_dir,
            "object_detection",
            save=True,
            train=True,
            width=args.image_width,
            height=args.image_height,
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
                                    num_workers=args.num_workers,
                                    collate_fn=lambda x: tuple(zip(*x))),
                "val": DataLoader(val_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda x: tuple(zip(*x)))
            }

            # model
            logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing model...')
            model = InsectDetector(num_classes=len(dataset.bollworms))
            model = model.to(device)

            # loss function
            logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing loss function...')
            # criterion = None
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
            model = train_object_detector(model, args.learning_rate, dataloaders, device, num_epochs=args.epochs, kfold_idx=kfold_idx, num_folds=args.splits, logger=run)

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
            
    if initial_args.do_image_classification:
        sweep_run_name = f"{datetime.datetime.now().strftime(f'%H-%M-%ST%d-%m-%Y')}_{generate_random_string(5)}"

        # directory to save models and parameters
        results_dir = get_dir(f'{initial_args.output_dir}/ic/{sweep_run_name}')

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = WadhwaniBollwormDataset(
            args.data_dir,
            "classification",
            save=True,
            train=True,
            width=args.image_width,
            height=args.image_height,
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
                                    num_workers=args.num_workers,
                                    collate_fn=lambda x: tuple(zip(*x))),
                "val": DataLoader(val_ds,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=lambda x: tuple(zip(*x)))
            }

            # model
            logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing model...')
            model = ImageClassifier(num_classes=len(dataset.classes))
            model = model.to(device)

            # loss function
            logging.info(f'Fold {kfold_idx + 1} / {args.splits}: preparing loss function...')
            criterion = nn.CrossEntropyLoss()
            criterion.to(device)

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
            model = train_classifier(model, args.learning_rate, dataloaders, device, num_epochs=args.epochs, kfold_idx=kfold_idx, num_folds=args.splits, logger=run)

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
