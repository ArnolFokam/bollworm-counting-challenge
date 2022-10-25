from typing import Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from bow.dataset import WadhwaniBollwormDataset
from bow.transform import BaselineTrainTransform


class ModelMixin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=10**(-4))
            if module.bias is not None:
                module.bias.data.zero_()
        

class InsectDetector(ModelMixin):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        # get number of input features for the classifier
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
    def forward(self, images: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone(images, targets)
        

if __name__ == '__main__':
    ds = WadhwaniBollwormDataset('data', transform=BaselineTrainTransform())
    
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    loader = iter(loader)
    imgs, targets = next(loader) 
    
    # modelling
    model = InsectDetector(num_classes=len(ds.bollworms))
    out = model(imgs, targets)   
