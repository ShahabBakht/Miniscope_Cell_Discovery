import random
import numbers
import math
import collections
import numpy as np
from PIL import ImageOps, Image
from joblib import Parallel, delayed

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]