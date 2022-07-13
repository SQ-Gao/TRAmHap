import argparse
import os,sys,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

plt.switch_backend('agg')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(1234)
