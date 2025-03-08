import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
import os
from test import test_model


test_model("mobilenet")
test_model("resnet")