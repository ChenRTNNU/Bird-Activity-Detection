import os
# import random
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
# import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.optim as optim
# from scipy.ndimage import median_filter
# import matplotlib.pyplot as plt
# from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

# from baseline.Evaluate import train, valid, eval_acc, eval_auc
from utils import make_sure_path_exists, show_f1score, show_loss

import importlib
import argparse


def main():
    ROOT=os.getcwd()
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Dynamic module import based on runtime parameter.')
    parser.add_argument('--aug', default='baseline' , type=str, required=True, help='The augmentation module to import from.')
    parser.add_argument('--repeat', default=0, type=int, required=True, help='repeat')
    args = parser.parse_args()
    savpath = os.path.join(ROOT, 'result',args.aug)
    module_name = f"{args.aug}.{args.aug}"
    module = importlib.import_module(module_name)
    run = getattr(module, 'run')
    run(args.repeat,savpath)
if __name__ == '__main__':
    main()