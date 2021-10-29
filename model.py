import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

data = pd.read_csv('breast-cancer-wisconsin.data')
print(data)

#todo - fjerne outliers fra dataset

# check wether cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
