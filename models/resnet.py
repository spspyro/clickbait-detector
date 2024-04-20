import torch
from torch import nn
from torchvision import models

__all__ = ['resnet50']

def get_lastlayer_channel(model):
    dummy = torch.rand((1, 3, 512, 512))
    model.eval()
    out_tensor = model(dummy)
    return out_tensor.shape[1]

def resnet50(
        num_class,
        pretrained=True,
        feature_extractor=True,
        progress=True):
    return ResNet50(num_class, pretrained, feature_extractor, progress)

class ResNet50(nn.Module):
    def __init__(
            self,
            num_class,
            pretrained=True,
            feature_extractor=True,
            progress=True):
        '''
        ResNet50 model class.
        Args:
            num_class:
                num_class for classification
            pretrained:
                pretrained the model
            feature_extractor:
                freeze backbone as feature extractor
            progress:
                display progress
        '''
        super().__init__()
        # If feature extractor, backbone has to be pretrained
        if feature_extractor:
            pretrained=True
        # Extracting backbone from torchvision resnet50
        resnet50 = models.resnet50(pretrained, progress)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        # Freeze backbone if feature extractor
        if feature_extractor:
            self.freeze_backbone()
        # Get out channel from backbone
        output_channel = get_lastlayer_channel(self.backbone)
        # Initialize head and loss module
        self.cls_head = nn.Linear(output_channel, num_class)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def freeze_backbone(self):
        '''Freeze the backbone as feature extractor.'''
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, X, y=None):
        '''Forward function for ResNet50.'''
        x = self.backbone(X)
        x = torch.flatten(x, 1)
        pred = self.cls_head(x)
        if self.training:
            assert(y is not None)
            loss = self.loss(pred, y)
            return loss
        else:
            return pred





















