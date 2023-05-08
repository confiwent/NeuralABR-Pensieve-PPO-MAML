import argparse, time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import logging, os
from train_a2c import train_a2c
from maml_train_vmaf import train_maml_ppo
import config.args_maml as args_maml
# from model_ppo_torch import Actor, Critic
from test_vmaf import test
import envs.env_vmaf as env
import envs.fixed_env_vmaf as env_test
import utils.load_trace as load_trace

S_INFO = 7 # 
S_LEN = 8 # maximum length of states 
C_LEN = 0 # content length 
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
TOTAL_CHUNK_NUM = 49
QUALITY_PENALTY = 0.8469011 #dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887

LOG_FILE = './variant_vmaf/Results/test/a2c/'
TEST_MODEL = './model/ppo/abr_ppo_92000.model'
TEST_TRACES = './test_traces/'
TRAIN_TRACES = './cooked_traces/'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def get_test_traces():
    log_save_dir = LOG_FILE
    test_traces = TEST_TRACES
    log_path = log_save_dir + 'log_test_a2c' #args.name
    return log_save_dir, test_traces, log_path

def run_test(args, video_vmaf_file, video_size_file):
    log_save_dir, test_traces, log_path = get_test_traces() 
    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)
    test_model_ = TEST_MODEL
    
    # video_size_file = './envs/video_size/Avengers/video_size_'

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                    all_cooked_bw=all_cooked_bw, 
                                    all_file_names = all_file_names, 
                                    video_size_file = video_size_file, 
                                    video_psnr_file= video_vmaf_file
                                    )
    test_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                            QUALITY_PENALTY, REBUF_PENALTY, \
                            SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)
    
    test(args, test_model_, test_env, log_path, log_save_dir)

def main():
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    parser = argparse.ArgumentParser()
    _, rest_args = parser.parse_known_args() 
    args = args_maml.get_args(rest_args)
    video_size_file = './video_size/ori/video_size_' #video = 'origin'
    video_vmaf_file = './variant_vmaf/video_vmaf/chunk_vmaf'
    args.ckpt = f'{ts}'

    if args.test:
        run_test(args, video_vmaf_file, video_size_file)
    else:
        if torch.cuda.is_available():
                torch.cuda.set_device(0) # ID of GPU to be used
                print("CUDA Device: %d" %torch.cuda.current_device())
        # -------- load envs ---
        Train_traces = TRAIN_TRACES
        Valid_traces = TEST_TRACES
        all_cooked_time, all_cooked_bw, all_file_names = \
                                    load_trace.load_trace(Valid_traces)
        valid_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, 
                                all_file_names = all_file_names, 
                                video_size_file = video_size_file, 
                                video_psnr_file= video_vmaf_file
                                )
        valid_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                    QUALITY_PENALTY, REBUF_PENALTY, SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)
        
        all_cooked_time, all_cooked_bw, all_file_names = \
                                            load_trace.load_trace(Train_traces)
        
        train_env = [env.Environment(
                            all_cooked_time=all_cooked_time, 
                            all_cooked_bw=all_cooked_bw,
                            all_file_names = all_file_names, 
                            video_size_file= video_size_file,
                            video_psnr_file= video_vmaf_file) \
                                for _ in range(args.agent_num)]
        for _ in range(args.agent_num):
            train_env[_].set_env_info(S_INFO, S_LEN, C_LEN, 
                        TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 
                        QUALITY_PENALTY, REBUF_PENALTY, 
                        SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)

        if args.a2c:
            train_a2c(args, train_env, valid_env)
        elif args.ppo:
            # train_ppo()
            print('to be continue...')
        elif args.a2br:
            train_env_ = train_env[0]
            del train_env
            train_maml_ppo(args, train_env_, valid_env)
            

if __name__ == '__main__':
    main()