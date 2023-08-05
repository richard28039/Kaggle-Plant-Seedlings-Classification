import torch 
import torch.nn as nn 


class CustomizedResNet(torch.nn.Module):
    def __init__(self, backbone, num_classes=0, freeze_layers=None):
        super().__init__()
        self.backbone = backbone

        if freeze_layers is not None:
            self.freeze_layers(freeze_layers)

        self.fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return x
    
    def freeze_layers(self, num_layers):
        for i, param in enumerate(self.backbone.parameters()):
            if i in num_layers:
                param.requires_grad = False
