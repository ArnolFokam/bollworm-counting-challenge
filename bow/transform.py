from typing import Dict, List, Union

import scipy
import torch
import numpy as np
import albumentations as A
from PIL.Image import Image
import albumentations.pytorch.transforms as TorchT


class BaselineTrainTransform:
    def __init__(self, train: bool = True) -> None:
        self.train = train
        
        if self.train:
            
            self.transform = A.Compose([
                A.RandomCrop(width=450, height=450),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                TorchT.ToTensorV2(p=1.0) ,
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
        else:
            return A.Compose([TorchT.ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
            

    def __call__(self, image: Union[np.ndarray, Image], bboxes: List[float], class_labels: List[str]) -> Dict[str,  Union[List[float], Image]]:
        transformed = self.transform(image=image, bboxes=bboxes,  class_labels=class_labels)
        transformed_class_labels = transformed['class_labels']
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        return transformed_image, transformed_bboxes, transformed_class_labels
        

if __name__ == '__main__':
    pass
