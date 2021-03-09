import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import logging
from train import train
# from model_ppo_torch import Actor, Critic
# from test_a3c_torch import valid, test
# import env as Env

LOG_FILE = './Results/sim/a3c/log'
TEST_PATH = './models/A3C/BC/360_a3c_240000.model'

parser = argparse.ArgumentParser(description='PPO_ABRori')
parser.add_argument('--test', action='store_true', help='Evaluate only')

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def main():
    args = parser.parse_args()
    if args.test:
        test(TEST_PATH)
    else:
        if torch.cuda.is_available():
                torch.cuda.set_device(0) # ID of GPU to be used
                print("CUDA Device: %d" %torch.cuda.current_device())
        train()

if __name__ == '__main__':
    main()