import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from model_torch import ActorCritic
from test_torch import valid, test
import env
import load_trace
from replay_memory import ReplayMemory

RANDOM_SEED = 52
S_INFO = 6
S_LEN = 8
A_DIM = 6
LEARNING_RATE = 0.0001
# TRAIN_SEQ_LEN = 100  # take as a train batch
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
# RANDOM_SEED = 42
# GAMMA = 0.90
# ENTROPY_WEIGHT = 0.99
UPDATE_INTERVAL = 1000
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
SUMMARY_DIR = './Results/sim/ac'
LOG_FILE = './Results/sim/ac/log'
# TEST_PATH = './models/A3C/BC/360_a3c_240000.model'

parser = argparse.ArgumentParser(description='ac_pytorch')
parser.add_argument('--test', action='store_true', help='Evaluate only')

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train_ac():
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    
    with open(LOG_FILE + '_record', 'w') as log_file, open(LOG_FILE + '_test', 'w') as test_log_file:
        # entropy_weight = ENTROPY_WEIGHT
        # value_loss_coef = 0.5
        torch.manual_seed(RANDOM_SEED)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

        model_ac = ActorCritic(A_DIM).type(dtype)

        model_ac.train()

        optimizer = optim.Adam(model_ac.parameters(), lr=LEARNING_RATE)

        # max_grad_norm = MAX_GRAD_NORM 

        state = np.zeros((S_INFO,S_LEN))
        state = torch.from_numpy(state)
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        # action_vec = np.zeros(A_DIM)
        # action_vec[bit_rate] = 1

        done = True
        epoch = 0
        time_stamp = 0

        exploration_size = 16
        episode_steps = 20
        update_num = 1
        batch_size = exploration_size * episode_steps #64
        gamma = 0.95
        gae_param = 0.90
        ent_coeff = 2.6
        cl_coeff = 0.2
        memory = ReplayMemory(exploration_size * episode_steps)
        # memory = ReplayMemory()

        while True:

            for explore in range(exploration_size):
                states = []
                actions = []
                rewards_comparison = []
                rewards = []
                values = []
                returns = []
                advantages = []

                for step in range(episode_steps):

                    prob, v = model_ac(state.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    v = v.detach().cpu()
                    # seed_ = np.random.uniform(0,1)
                    # if np.random.uniform(0,1) <= exploration_threhold:
                    #     action = random.randint(0, 5)
                    #     action = torch.tensor([[action]]).type(dlongtype)
                    # else:
                    #     action = prob.multinomial(num_samples=1)
                    values.append(v)

                    bit_rate = int(action.squeeze().cpu().numpy())

                    actions.append(torch.tensor([action]))
                    states.append(state.unsqueeze(0))

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
                    reward_max = 2.67
                    reward = float(max(min(reward, reward_max), -10*reward_max) / reward_max)
                    rewards.append(reward)
                    rewards_comparison.append(torch.tensor([reward]))

                    last_bit_rate = bit_rate

                    # retrieve previous state
                    if end_of_video:
                        state = np.zeros((S_INFO, S_LEN))
                        state = torch.from_numpy(state)
                        last_bit_rate = DEFAULT_QUALITY
                        break

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

                    # log time_stamp, bit_rate, buffer_size, reward
                    log_file.write(str(time_stamp) + '\t' +
                                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                                str(buffer_size) + '\t' +
                                str(rebuf) + '\t' +
                                str(video_chunk_size) + '\t' +
                                str(delay) + '\t' +
                                str(reward) + '\n')
                    log_file.flush()

                # one last step
                R = torch.zeros(1, 1)
                if end_of_video == False:
                    _, v = model_ac(state.unsqueeze(0).type(dtype))
                    v = v.detach().cpu()
                    R = v.data
                #================================结束一个ep========================================
                # compute returns and GAE(lambda) advantages:
                values.append(Variable(R))
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    A = float(td) + gamma * gae_param * A
                    advantages.insert(0, A)
                    R = A + values[i]
                    returns.insert(0, R)
                # store usefull info:
                # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
        
            # policy grad updates:
            model_ac.zero_grad()

            # mini_batch
            batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)
            probs_pre, values_pre = model_ac(batch_states.type(dtype))

            # actor_loss
            prob_value = torch.gather(probs_pre, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))
            policy_loss = -torch.mean(prob_value * batch_advantages.type(dtype))
            loss_ent = ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))
            actor_loss = policy_loss + loss_ent

            # critic_loss
            vf_loss = (values_pre - batch_returns.type(dtype)) ** 2 # V_\theta - Q'
            critic_loss = cl_coeff * 0.5 * torch.mean(vf_loss)

            # update
            total_loss = critic_loss + policy_loss + loss_ent
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            ## test and save the model
            epoch += 1
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(policy_loss.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(critic_loss.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(loss_ent.detach().cpu().numpy()))

            if epoch % UPDATE_INTERVAL == 0:
                logging.info("Model saved in file")
                valid(model_ac, epoch, test_log_file, SUMMARY_DIR, 'ac')
                # entropy_weight = 0.95 * entropy_weight
                ent_coeff = 0.95 * ent_coeff
# def main():
#     train_ac()

# if __name__ == '__main__':
#     main()