import time
from collections import deque
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model_ac_torch import Actor, Critic
import env as env_valid
import fixed_env as env_test
import load_trace

RANDOM_SEED = 42
S_INFO = 6
S_LEN = 8
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
TEST_TRACES_VALID = './cooked_test_traces/'
SUMMARY_DIR = './Results/sim/a2c'
LOG_FILE = './Results/sim/a2c/log'
TEST_LOG_FOLDER = './Results/sim/a2c/test_results/'

LOG_FILE_VALID = './Results/sim/a2c/test_results/log_valid_a2c'
TEST_LOG_FOLDER_VALID = './Results/sim/a2c/test_results/'

# LOG_FILE_TEST = './Results/test/BC/log_hybrid_ppo'
# SUMMARY_DIR = './Results/test/BC/'

Log_path = './Results/sim/a2c'

def evaluation(model, log_path_ini, net_env, all_file_name, detail_log = True):

    # all_file_name = net_env.get_file_name()

    state = np.zeros((S_INFO,S_LEN))
    state = torch.from_numpy(state)
    # reward_sum = 0
    done = True
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    # model.load_state_dict(model.state_dict())
    log_path = log_path_ini + '_' + all_file_name[net_env.trace_idx]
    log_file = open(log_path, 'w')
    time_stamp = 0
    for video_count in tqdm(range(len(all_file_name))):
        while True:
            with torch.no_grad():
                prob= model(state.unsqueeze(0).type(torch.FloatTensor))
            action = prob.multinomial(num_samples=1).detach()
            bit_rate = int(action.squeeze().cpu().numpy())

            if detail_log == False:
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain = \
                        net_env.get_video_chunk(bit_rate) ## sample in the environment of virtual player
                
                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # reward is video quality - rebuffer penalty - smooth penalty
                # -- lin scale reward --
                # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                #         - REBUF_PENALTY * rebuf \
                #         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                #                                 VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                # -- log scale reward --
                log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
                log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

                reward = log_bit_rate \
                        - REBUF_PENALTY * rebuf \
                        - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
                
                last_bit_rate = bit_rate


                log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
                log_file.flush()

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                state = torch.from_numpy(state)
            
            if end_of_video:
                state = np.zeros((S_INFO,S_LEN))
                state = torch.from_numpy(state)
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                log_file.write('\n')
                log_file.close()
                time_stamp = 0

                if video_count >= len(all_file_name):
                    break
                else:
                    log_path = LOG_FILE_VALID + '_' + all_file_name[net_env.trace_idx]
                    log_file = open(log_path, 'w')
                    break
            # else:
                # delay, video_chunk_size, video_distortion, reward, next_state, done, rebuf, download_interval = env.step([int(action_1.squeeze()), int(action_2.squeeze()), int(A_DIM-1)])  # Step

                # # r_batch.append(reward)
                # time_stamp += download_interval

                # # log time_stamp, bit_rate, buffer_size, reward
                # log_file.write(str(time_stamp) + '\t' + 
                #             str(next_state[0][26]) + '\t' + str(next_state[0][27]) + '\t' + str(next_state[0][28]) + '\t' + # qualities
                #             str(next_state[0][8]*5.0) + '\t' + ## buffer size
                #             str(rebuf) + '\t' + ## rebuffering time
                #             str(reward) + '\t' + # chunk reward 
                #             str(video_distortion) + '\t' + 
                #             str(video_chunk_size) + '\t' + 
                #             str(delay) + '\n')
                # log_file.flush()

            # a quick hack to prevent the agent from stucking
            # actions.append(action[0, 0])
            # if actions.count(actions[0]) == actions.maxlen:
            #     done = True

            # if done:
            #     log_file.write('\n')
            #     log_file.close()
            #     state = np.zeros((1,S_LEN))
            #     time_stamp = 0

            #     if video_count >= len(all_file_name):
            #         break
            #     else:
            #         log_path = LOG_FILE_VALID + '_' + all_file_name[env.trace_idx]
            #         log_file = open(log_path, 'w')
            #         break

            # state = torch.from_numpy(state)

def valid(shared_model, epoch, log_file):
                
    os.system('rm -r ' + TEST_LOG_FOLDER_VALID)
    os.system('mkdir ' + TEST_LOG_FOLDER_VALID)

    model = Actor(A_DIM).type(torch.FloatTensor)
    model.eval()
    model.load_state_dict(shared_model.state_dict())
    log_path_ini = LOG_FILE_VALID
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES_VALID)
    env = env_valid.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)
    evaluation(model, log_path_ini, env, all_file_names, False)

    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(int(epoch)) + '\t' +
                str(rewards_min) + '\t' +
                str(rewards_5per) + '\t' +
                str(rewards_mean) + '\t' +
                str(rewards_median) + '\t' +
                str(rewards_95per) + '\t' +
                str(rewards_max) + '\n')
    log_file.flush()

    add_str = 'a2c'
    model_save_path = Log_path + "/%s_%s_%d.model" %(str('abr'), add_str, int(epoch))
    torch.save(shared_model.state_dict(), model_save_path)

def test(test_path):
    env = env_test.Environment()

    model = Actor(A_DIM).type(torch.FloatTensor)
    model.eval()
    model.load_state_dict(torch.load(test_path))
    log_path_ini = LOG_FILE_TEST
    env = env_valid.Environment()
    evaluation(model, log_path_ini, env, True)
