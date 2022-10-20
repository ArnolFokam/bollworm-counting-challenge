import os
import glob
import logging
from PIL import Image
from typing import List
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Callable, Optional

from bow.helpers import get_dir


load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s:', datefmt='%H:%M:%S')


class WadhwaniBollwormDataset(torch.utils.data.Dataset):
    
    bbox_path = "images_bboxes.csv"
    images_path = "images"
    
    bollworms = ["abw", "pbw"]


    def __init__(
            self,
            root_dir: str,
            train: bool = True,
            download: bool = False,
            transform: Optional[Callable] = None,
            max_cache_length: int = 256):
        
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        self.max_cache_length = max_cache_length
        self.class_meta = self.__load_meta_class()
        
        # get the ids of the image in the folder
        self.image_ids = [os.path.basename(filename).split(".")[0].split("_")[-1] for filename in glob.glob(f"{root_dir}/{self.images_path}/id_*.jpg")]
        
        # get the bounding boxes
        if self.train:
            self.bboxes = []
            self.targets = []
            
            bboxes_df = pd.read_csv(f"{self.root_dir}/{self.bbox_path}")
            
            for name, bboxes in bboxes_df.groupby("image_id"):
                targets = []
                bboxes = []
                for _, row in bboxes.iterrows():
                    targets.append(row['worm_type'])
                    bboxes.append(row["geometry"].bounds)
            
        
        # save cache for our load on the fly algorithm
        self.cache = {}

        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: str):
        image_id, bbox, target = self.image_ids[index], self.bboxes[index]if self.train else torch.empty(1), self.targets[index] if self.train else torch.empty(1)
        image_id = np.array(self.__get_image_from_id(image_id))
        if self.transform:
            transformed = self.transform(image=image.astype(np.float32))
            image = transformed["image"].float()
        else:
            image = torch.FloatTensor(image)

        return int(image_id), image, field_mask, int(self.class_meta[target]["loss_label"]) if self.train else torch.empty(1)

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
            with Image.open(f"{self.root_dir}/{self.images_path}/id_{image_id}.jpg") as im:
                
                # if max cache length attain, remnove one random element
                if len(self.cache.keys()) >= self.max_cache_length:
                    del self.cache[random.choice(self.cache.keys())]
                
                # insert next element
                self.cache[image_id] = im
                return self.cache[image_id]


if __name__ == '__main__':
    ds = WadhwaniBollwormDataset('data/', train=True)
    pass
