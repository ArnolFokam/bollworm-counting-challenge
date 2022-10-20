import numpy as np
from typing import Dict, List, Union
from PIL.Image import Image
import torch
import scipy
from aic.augmentation import RandomFieldAreaCrop, ReduceSkewness, RotateBands, NormalizeBands

import albumentations as A
import albumentations.pytorch.transforms as TorchT

from aic.dataset import AgriFieldDataset


class BaselineTrainTransform:
    def __init__(self, bands: List[str], vegetative_indeces: List[str], crop_size: int = 32) -> None:
        self.crop_size = crop_size
        self.bands = bands
        self.vegetative_indeces = vegetative_indeces
        
        # transform for RGB bands
        self.rgb_transform = A.Compose([
            A.HueSaturationValue(p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=.015)
        ])
        
        # transform that change the value of a voxel in the spectral bands
        self.voxel_value_transform = A.Compose([
            NormalizeBands(mean=[AgriFieldDataset.mean[band] for band in self.bands] + [AgriFieldDataset.mean[band] for band in self.vegetative_indeces], 
                           std=[AgriFieldDataset.std[band] for band in self.bands] + [AgriFieldDataset.mean[band] for band in self.vegetative_indeces]),
            ReduceSkewness(),
        ])

        # transform that changes the geometric shape of the image (rotation, translation, etc)
        self.geometric_transform = A.Compose([
            RandomFieldAreaCrop(crop_size=self.crop_size),
            A.Flip(),
            RotateBands(limit=180),    
        ])

        # transform after all the important ones, usually to convert to tensor
        self.final_transform = A.Compose([
            TorchT.ToTensorV2()
        ])

    def __call__(self, image: Union[np.ndarray, Image], mask: Union[np.ndarray, Image]) -> Dict[str,  Union[np.ndarray, Image]]:
        image[:, :, :3] = self.rgb_transform(image=image[:, :, :3])["image"]
        image = self.voxel_value_transform(image=image)["image"]
        
        # dilate mask to slight increase the region of interest
        mask = scipy.ndimage.binary_dilation(mask.astype(np.uint8), structure=np.ones((5, 5),np.uint8), iterations = 2).astype(np.float64)
        
        transformed = self.geometric_transform(image=image, mask=mask)
        transformed = self.final_transform(image=transformed["image"], mask=transformed["mask"])

        return {
            "image": transformed["image"],
            "mask": transformed["mask"]
        }
        
        
class BaselineEvalTransform(BaselineTrainTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # transform for RGB bands
        self.rgb_transform = A.Compose([])

        # transform that changes the geometric shape of the image (rotation, translation, etc)
        self.geometric_transform = A.Compose([
            RandomFieldAreaCrop(crop_size=self.crop_size), 
        ])
    

if __name__ == '__main__':
    bands = ['B01', 'B02', 'B03', 'B04','B05','B06','B07','B08','B8A', 'B09', 'B11', 'B12']
    ds = AgriFieldDataset('data/source', bands=bands, transform=BaselineTrainTransform(bands=bands, crop_size=16), train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    fids, imgs, masks, targets = next(loader)
