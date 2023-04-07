import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import logging, os
from train_ppo_gae import train_ppo
from train_a2c import train_a2c
# from model_ppo_torch import Actor, Critic
from test_vmaf import test
import env_vmaf as env
import fixed_env_vmaf as env_test
import load_trace

S_INFO = 7 # 
S_LEN = 6 # maximum length of states 
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

parser = argparse.ArgumentParser(description='RL-based ABR with vmaf')
parser.add_argument('--test', action='store_true', help='Evaluate only')
parser.add_argument('--name', default='pensieve', help='the name of algorithm')
parser.add_argument('--agent-num', nargs='?', const=16, default=16, type=int, help='env numbers')
parser.add_argument('--proba', action='store_true', help='Use probabilistic policy')
parser.add_argument('--a2c', action='store_true', help='Train policy with A2C')
parser.add_argument('--ppo', action='store_true', help='Train policy with PPO')

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def get_test_traces():
    # if args.tf:
    #     log_save_dir = TEST_LOG_FILE_FCC
    #     test_traces = TEST_TRACES_FCC
    # elif args.t3g:
    #     log_save_dir = TEST_LOG_FILE_3GP
    #     test_traces = TEST_TRACES_3GP
    # elif args.to:
    #     log_save_dir = TEST_LOG_FILE_OBE
    #     test_traces = TEST_TRACES_OBE
    # elif args.tg:
    #     log_save_dir = TEST_LOG_FILE_GHT
    #     test_traces = TEST_TRACES_GHT
    # elif args.tn:
    #     log_save_dir = TEST_LOG_FILE_FHN
    #     test_traces = TEST_TRACES_FHN
    # elif args.tp:
    #     log_save_dir = TEST_LOG_FILE_PUF
    #     test_traces = TEST_TRACES_PUF
    # elif args.tp2:
    #     log_save_dir = TEST_LOG_FILE_PUF2
    #     test_traces = TEST_TRACES_PUF2
    # elif args.tfh:
    #     log_save_dir = TEST_LOG_FILE_FH
    #     test_traces = TEST_TRACES_FH
    # elif args.tw:
    #     log_save_dir = TEST_LOG_FILE_PWI
    #     test_traces = TEST_TRACES_PWI
    # elif args.ti:
    #     log_save_dir = TEST_LOG_FILE_INT
    #     test_traces = TEST_TRACES_INT
    # else:
    #     # print("Please choose the throughput data traces!!!")
    #     log_save_dir = TEST_LOG_FILE_FCC
    #     test_traces = TEST_TRACES_FCC
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
                            QUALITY_PENALTY, REBUF_PENALTY, SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)
    
    test(args, test_model_, test_env, log_path, log_save_dir)

def main():
    # test(TEST_MODEL, TEST_TRACES, LOG_FILE)
    args = parser.parse_args()
    video_size_file = './video_size/ori/video_size_' #video = 'origin'
    video_vmaf_file = './variant_vmaf/video_vmaf/chunk_vmaf'

    if args.test:
        run_test(args, video_vmaf_file, video_size_file)
    else:
        if torch.cuda.is_available():
                torch.cuda.set_device(0) # ID of GPU to be used
                print("CUDA Device: %d" %torch.cuda.current_device())
        # -------- load envs ---
        Train_traces = TRAIN_TRACES
        Valid_traces = TEST_TRACES
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Valid_traces)
        valid_env = env_test.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, 
                                all_file_names = all_file_names, 
                                video_size_file = video_size_file, 
                                video_psnr_file= video_vmaf_file
                                )
        valid_env.set_env_info(S_INFO, S_LEN, C_LEN, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                    QUALITY_PENALTY, REBUF_PENALTY, SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)
        
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Train_traces)
        
        train_env = [env.Environment(
                            all_cooked_time=all_cooked_time, 
                            all_cooked_bw=all_cooked_bw, 
                            video_size_file= video_size_file,
                            video_psnr_file= video_vmaf_file) for _ in range(args.agent_num)]
        for _ in range(args.agent_num):
            train_env[_].set_env_info(S_INFO, S_LEN, C_LEN, 
                        TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, 
                        QUALITY_PENALTY, REBUF_PENALTY, SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)

        if args.a2c:
            train_a2c(args, train_env, valid_env)
        elif args.ppo:
            # train_ppo()
            print('to be continue...')

if __name__ == '__main__':
    main()