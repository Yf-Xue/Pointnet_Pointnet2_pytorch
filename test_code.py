import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from pytorch3d import ops
pcd = torch.randn(5, 128, 1,3)
points1 = torch.randn(5, 1, 32, 3)
pcd = pcd.permute(0,3,1,2)
points1 = points1.permute(0,3,1,2)
edist = torch.pairwise_distance(points1, pcd, p=2)
print(edist.size())