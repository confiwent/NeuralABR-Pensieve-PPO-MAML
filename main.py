import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import logging, argparse
from train_ppo_gae import train_ppo
from train_ac import train_ac
from train_a2c import train_a2c
from train_maml import train_maml_ppo

# from model_ppo_torch import Actor, Critic
from test_ppo_torch import valid, test
import variant_vmaf.config.args_maml as args_maml

# import env as Env

LOG_FILE = "./Results/test/"
TEST_MODEL = "./model/ppo/abr_ppo_92000.model"
TEST_TRACES = "./test_traces/"
# TEST_TRACES = './test/'

parser = argparse.ArgumentParser(description="RL-based ABR")
parser.add_argument("--test", action="store_true", help="Evaluate only")
parser.add_argument("--a2c", action="store_true", help="Train policy with A2C")
parser.add_argument("--ppo", action="store_true", help="Train policy with PPO")
parser.add_argument("--maml", action="store_true", help="Train policy with MAML")
parser.add_argument("--lin", action="store_true", help="Using Linear metric")

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def main():
    # test(TEST_MODEL, TEST_TRACES, LOG_FILE)
    args, rest_args = parser.parse_known_args()

    # parser_others = argparse.ArgumentParser()
    # _, rest_args = parser_others.parse_known_args()
    args_others = args_maml.get_args(rest_args)
    if args.lin:
        args_others.lin = True
    else:
        args_others.lin = False
    if args.test:
        test(TEST_MODEL, TEST_TRACES, LOG_FILE)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # ID of GPU to be used
            print("CUDA Device: %d" % torch.cuda.current_device())

        if args.a2c:
            train_a2c()
        elif args.ppo:
            train_ppo()
        elif args.maml:
            train_maml_ppo(args_others)


if __name__ == "__main__":
    main()
