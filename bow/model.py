import torch
import torch.nn as nn

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
    def __init__(self):
        super().__init__()
        pass
    
    def forward():
        raise NotImplementedError
        

if __name__ == '__main__':
    ds = WadhwaniBollwormDataset('data', transform=BaselineTrainTransform())
    
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    loader = iter(loader)
    ids, imgs, bboxes, targets = next(loader) 
    
    # modelling
    model = InsectDetector()
    out = model(imgs, masks)   
