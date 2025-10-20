import torch.nn as nn
from torchvision import models
from typing import Iterable

from .multihead import MultiHeadClassifier


class VIT(nn.Module):
    def __init__(self, num_classes:  int | Iterable[int], freeze: bool = True):
        super(VIT, self).__init__()

        self.latent_dim = 256
        self.num_classes = num_classes if isinstance(
            num_classes, Iterable) else (num_classes,)

        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # self.vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        # self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        # self.vit = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
        # self.vit = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)

        # self.vit.heads = nn.Identity()  # remove existing classification head

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        # self.classifiers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(768, self.latent_dim),
        #         nn.ReLU(),
        #         nn.Linear(self.latent_dim, num_classes),
        #     ) for num_classes in self.num_classes
        # ])

        self.vit.heads = MultiHeadClassifier(768, *num_classes)

    def forward(self, x):
        return self.vit(x)
