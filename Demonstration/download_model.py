import json
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusers import StableDiffusionModelEditingPipeline
from tw_rouge import get_rouge
import os
import argparse
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

# Whether torch can use GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Training Device: " + device)

model = StableDiffusionModelEditingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
model.save_pretrained("./pretrained_model")

