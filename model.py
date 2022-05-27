from typing import Dict
import numpy as np
import torch
from torch import nn
from transformers import RobertaModel, AutoModel
import torchvision

class KoCLIP(nn.Module):
    def __init__(self, pvm: str, embed_dim: int):
        super().__init__()
        self.pvm = pvm
        if self.pvm=='RN101':
            self.visual_model = torchvision.models.resnet101(pretrained=True)
            final_dim = 1000
        else:
            self.visual_model = AutoModel.from_pretrained(self.pvm)
            final_dim = self.visual_model.config.hidden_size
        self.language_model = RobertaModel.from_pretrained('klue/roberta-large')
        self.image_projection = nn.Linear(final_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(self.language_model.config.hidden_size, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        if self.pvm=='RN101':
            x = self.visual_model(image)
        else:
            x = self.visual_model(image).pooler_output
        x = self.image_projection(x)
        return x

    def encode_text(self, text: Dict):
        x = self.language_model(**text).pooler_output
        x = self.text_projection(x)
        return x

    def forward(self, image, text: Dict):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features