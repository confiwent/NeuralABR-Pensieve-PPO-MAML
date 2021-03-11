import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from model_ac_torch import Actor, Critic
from test_ppo_torch import valid, test
import env
import load_trace
from replay_memory import ReplayMemory

RANDOM_SEED = 28
S_INFO = 6
S_LEN = 8
A_DIM = 6
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
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
MIN_MOMERY = 100
SUMMARY_DIR = './Results/sim/ppo'
LOG_FILE = './Results/sim/ppo/log'
# TEST_PATH = './models/A3C/BC/360_a3c_240000.model'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train_ppo():
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

        model_actor = Actor(A_DIM).type(dtype)
        model_critic = Critic(A_DIM).type(dtype)

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.RMSprop(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.RMSprop(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

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

        exploration_size = 10
        episode_steps = 25
        update_num = 1
        batch_size = 128
        gamma = 0.99
        gae_param = 0.95
        clip = 0.02 # 0.02 works
        ent_coeff = 0.98
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

                    prob = model_actor(state.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
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
                    reward = float(max(min(reward, reward_max), -2*reward_max) / reward_max)
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
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                    # v = v.detach().cpu()
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
                    # R = gamma * R + rewards[i]
                    returns.insert(0, R)
                # store usefull info:
                # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                memory.push([states, actions, returns, advantages])
        
            # policy grad updates:
            model_actor_old = Actor(A_DIM).type(dtype)
            model_actor_old.load_state_dict(model_actor.state_dict())
            model_critic_old = Critic(A_DIM).type(dtype)
            model_critic_old.load_state_dict(model_critic.state_dict())

            ## actor update
            for update_step in range(update_num):
                model_actor.zero_grad()
                model_critic.zero_grad()

                # new mini_batch
                # priority_batch_size = int(memory.get_capacity()/10)
                batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)
                # batch_size = memory.return_size()
                # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)

                # old_prob
                probs_old = model_actor_old(batch_states.type(dtype).detach())
                v_pre_old = model_critic_old(batch_states.type(dtype).detach())
                prob_value_old = torch.gather(probs_old, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))

                # new prob
                probs = model_actor(batch_states.type(dtype))
                v_pre = model_critic(batch_states.type(dtype))
                prob_value = torch.gather(probs, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))

                # ratio
                ratio = prob_value / (1e-6 + prob_value_old)

                ## non-clip loss
                # surrogate_loss = ratio * batch_advantages.type(dtype)


                # clip loss
                surr1 = ratio * batch_advantages.type(dtype)  # surrogate from conservative policy iteration
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages.type(dtype)
                loss_clip_actor = -torch.mean(torch.min(surr1, surr2))
                # value loss
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
                # entropy
                loss_ent = ent_coeff * torch.mean(probs * torch.log(probs + 1e-6))
                # total
                policy_total_loss = loss_clip_actor + loss_ent
                # print("hell0")

                # copy the new model to old model?
                # model_actor_old.load_state_dict(model_actor.state_dict())
                # model_critic_old.load_state_dict(model_critic.state_dict())

                # update 
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                policy_total_loss.backward()
                # loss_clip_actor.backward(retain_graph=True)
                loss_value.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                # print("hell0")

            ## test and save the model
            epoch += 1
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(loss_clip_actor.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))

            if epoch % UPDATE_INTERVAL == 0:
                logging.info("Model saved in file")
                valid(model_actor, epoch, test_log_file)
                # entropy_weight = 0.95 * entropy_weight
                ent_coeff = 0.95 * ent_coeff
def main():
    train_ppo()

if __name__ == '__main__':
    main()