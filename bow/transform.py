import numpy as np
from typing import Dict, List, Union
from PIL.Image import Image
import torch
import scipy


class BaselineTrainTransform:
    def __init__(self,) -> None:
        pass

    def __call__(self, image: Union[np.ndarray, Image], bbox: List[float]) -> Dict[str,  Union[List[float], Image]]:
        raise NotImplementedError
        
        
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
