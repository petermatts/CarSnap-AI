import torch.nn as nn
from torchvision import models


class VIT(nn.Module):
    def __init__(self, num_classes: int):
        super(VIT, self).__init__()

        self.vit = models.vit_b_16(pretrained=True)
        # self.vit = models.vit_b_32(pretrained=True)
        # self.vit = models.vit_l_16(pretrained=True)
        # self.vit = models.vit_l_32(pretrained=True)
        # self.vit = models.vit_h_14(pretrained=True)

        self.vit.heads = nn.Identity()  # remove existing classification head

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.vit(x)
        out = self.classifier(features)
        return out
