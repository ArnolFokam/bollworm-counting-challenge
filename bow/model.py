from typing import Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

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

        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        # backbone = torchvision.models.swin_t(weights="DEFAULT").features

        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        self.model = FasterRCNN(backbone,
                                num_classes=num_classes,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)

    def forward(self, images: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(images, targets)


if __name__ == '__main__':
    ds = WadhwaniBollwormDataset('data', transform=BaselineTrainTransform())

    loader = torch.utils.data.DataLoader(
        ds, batch_size=20, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    loader = iter(loader)
    imgs, targets = next(loader)

    # modelling
    model = InsectDetector(num_classes=len(ds.bollworms))
    out = model(imgs, targets)
