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

if __name__ == '__main__':
    pass
