import torch
from torchvision import transforms
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from processing import normalise_zero_one

def test_normalise_zero_one():
    x1 = (torch.rand(4,3,96,192) -0.5)/0.5
    x2 = (torch.rand(4,3,96,192) -0.5)/0.5
    if x1.min() < 0 and x2.min() < 0:
        y = normalise_zero_one(x1,x2)
        assert y[0].min()>=0 and y[1].min()>= 0
