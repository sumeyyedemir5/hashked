import sys
import os
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIX2VOX_MODEL_DIR = os.path.join(PROJECT_ROOT, "Pix2Vox", "models")
sys.path.insert(0, PIX2VOX_MODEL_DIR)

from encoder import Encoder
from decoder import Decoder
from merger import Merger

class Pix2VoxA(nn.Module):
    def __init__(self, cfg):
        super(Pix2VoxA, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.merger = Merger(cfg)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        merged = self.merger(decoded)
        return merged
