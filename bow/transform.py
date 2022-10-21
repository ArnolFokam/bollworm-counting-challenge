from typing import Dict, List, Union

import numpy as np
import scipy
import torch
from PIL.Image import Image
import albumentations as A


class BaselineTrainTransform:
    def __init__(self,) -> None:
        self.transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image: Union[np.ndarray, Image], bboxes: List[float], class_labels: List[str]) -> Dict[str,  Union[List[float], Image]]:
        transfromed = self.transform(image=image, bboxes=bboxes,  class_labels=class_labels)
        transformed_class_labels = transformed['class_labels']
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        return transformed_image, transformed_bboxes, transformed_class_labels
        

if __name__ == '__main__':
    pass
