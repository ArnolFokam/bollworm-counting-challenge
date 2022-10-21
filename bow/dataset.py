import glob
import logging
import math
import os
import pickle
from collections import defaultdict
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd
import shapely.wkt
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from bow.helpers import get_dir

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')


class WadhwaniBollwormDataset(torch.utils.data.Dataset):
    
    bbox_path = "images_bboxes_uncorrupted.csv"
    images_path = "images"
    
    bollworms = ["abw", "pbw"]


    def __init__(
            self,
            root_dir: str,
            width: int = 256,
            height: int = 256,
            save: bool = True,
            train: bool = True,
            transform: Optional[Callable] = None,
            max_cache_length: int = 256):
        
        self.train = train
        
        # image size
        self.width = width
        self.height = height
        
        self.root_dir = root_dir
        self.transform = transform
        self.max_cache_length = max_cache_length
        self.class_meta = self.__load_meta_class()
        
        try:
            
            logging.info('Loading data from cache...')
            
            if self.train:
                with open(f'{self.root_dir}/cache_{self.bbox_path.split(".")[0]}.cache.pkl', 'rb') as f:
                    self.bboxes = pickle.load(f)
                    
            logging.info('Loaded!')
                    
        except (IOError, EOFError, OSError, pickle.PickleError, pickle.UnpicklingError):
            logging.info('Error occured during cache data loading. Preprocessing data again...')
            
            # get the bounding boxes
            self.bboxes = []
            self.targets = []
            
            bboxes_df = pd.read_csv(f"{self.root_dir}/{self.bbox_path}")
            
            pbar = tqdm(bboxes_df.groupby("image_id"))
            pbar.set_description("Getting bounding box information...")
            
            if self.train:
                for name, bboxes in pbar:
                    tmp_targets = []
                    tmp_bboxes = []
                    for _, row in bboxes.iterrows():
                        if not pd.isnull(row['worm_type']):
                            tmp_targets.append(row['worm_type'])
                            tmp_bboxes.append(shapely.wkt.loads(row["geometry"]).bounds)
            
                    self.bboxes.append((name.split(".")[0].split("_")[-1], tmp_bboxes, tmp_targets))
            
            if save:
                logging.info('Caching data for subsequent use...')

                with open(f'{self.root_dir}/cache_{self.bbox_path.split(".")[0]}.cache.pkl', 'wb') as f:
                    pickle.dump(self.bboxes, f)
                    
                logging.info('Cached!')

        
        # save cache for our load on the fly algorithm
        self.cache = {}

        
    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index: str):
        image_id, bboxes, targets = self.bboxes[index]
        
        # image
        image, original_height, original_width = self.__get_image_from_id(image_id)
        image /= 255.0
        
        # normalize the bounding boxes
        for i in range(len(bboxes)):
            xmin = (bboxes[i][0] / original_width) * self.width # min x
            ymin = (bboxes[i][1] / original_height) * self.height # max x
            xmax = (bboxes[i][2] / original_width) * self.width # min y
            ymax = (bboxes[i][3] / original_height) * self.height # max y
            
            bboxes[i] = (xmin, ymin, xmax, ymax)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        
        if self.transform:
            image, bboxes, targets = self.transform(image=image, bboxes=bboxes, class_labels=targets)
        
        # TODO: image id
        return image, {
            "image_id": torch.Tensor(index),
            "boxes": torch.Tensor(bboxes),
            "labels": torch.as_tensor([int(self.class_meta[target]["loss_label"]) for target in targets], dtype=torch.int64) if self.train else torch.empty(len(targets))
        }

    @staticmethod
    def get_class_weights(targets):
        raise NotImplementedError

    def __load_meta_class(self):
        """
        Returns a mapping of the true index 
        from the dataset to contiguous index 
        from 0 - 1 for classification loss
        """

        return {k: {
            "name": k,
            "loss_label": v,
            # can load more meta if you want
        } for k, v in zip(self.bollworms, range(len(self.bollworms)))}
        
    def __get_image_from_id(self, image_id: int):
        if image_id in self.cache.keys():
            # if image in cache, return the image
            return self.cache[image_id]
        else:
            image = cv2.imread(f"{self.root_dir}/{self.images_path}/id_{image_id}.jpg")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            original_height, original_width = img_rgb.shape[:2]
            img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
                
            # if max cache length attain, remnove one random element
            if len(self.cache.keys()) >= self.max_cache_length:
                del self.cache[random.choice(self.cache.keys())]
                
            # insert next element
            self.cache[image_id] = img_res, original_height, original_width
            return self.cache[image_id]


if __name__ == '__main__':
    ds = WadhwaniBollwormDataset('data/', train=True)
