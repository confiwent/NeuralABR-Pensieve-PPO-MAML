import os, pdb, datetime, json, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from models.model_ac_vmaf import Actor, Critic
from test_vmaf_v1 import valid
from utils.replay_memory import ReplayMemory
from utils.helper import save_models_a2c, clean_file_cache

RANDOM_SEED = 18
A_DIM = 6
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
DB_NORM_FACTOR = 100.0 
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
QUALITY_PENALTY = 0.8469011 #dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887
DEFAULT_QUALITY = 1  # default video quality without agent
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
SUMMARY_DIR = './variant_vmaf/Results/sim'

parser = argparse.ArgumentParser(description='a2c_pytorch')
parser.add_argument('--test', action='store_true', help='Evaluate only')

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train_a2c(args, train_env, valid_env):
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.name.split()))
    summary_dir = SUMMARY_DIR
    save_folder = os.path.join(*[summary_dir, net_desc])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)
    log_file_name = save_folder + '/log'

    logging.basicConfig(filename=log_file_name + '_central',
                        filemode='w',
                        level=logging.INFO)
    
    with open(log_file_name + '_record', 'w') as log_file,\
            open(log_file_name + '_test', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        s_info, s_len, _, _, _, quality_penalty, \
            rebuffer_penalty, smooth_penalty_p, smooth_penalty_n = \
                                                    train_env[0].get_env_info()

        model_actor = Actor(A_DIM).type(dtype)
        model_critic = Critic(A_DIM).type(dtype)

        if args.init:
            if not args.adp:
                init_ckpt_path = os.path.join(*[summary_dir, 'init_ckpt/a2c']) # Notice: ensure the correct model!
                # agent.load(init_ckpt_path)
                model_actor.load_state_dict(torch.load(init_ckpt_path + "/actor.model"))
                model_critic.load_state_dict(torch.load(init_ckpt_path + "/critic.model"))
                print("Initilization has done!")
            elif args.Ada2c:
                init_ckpt_path = os.path.join(*[summary_dir, 'adp/a2c_ini']) # Notice: ensure the correct model!
                # agent.load(init_ckpt_path)
                model_actor.load_state_dict(torch.load(init_ckpt_path + "/actor.model"))
                model_critic.load_state_dict(torch.load(init_ckpt_path + "/critic.model"))
                print("Initilization for A2C adaptation has done!")
            else:
                init_ckpt_path = os.path.join(*[summary_dir, 'adp/maml_ini']) # Notice: ensure the correct model!
                # agent.load(init_ckpt_path)
                model_actor.load_state_dict(torch.load(init_ckpt_path + "/actor_i.pt"))
                model_critic.load_state_dict(torch.load(init_ckpt_path + "/critic_i.pt"))
                print("Initilization for adaptation has done!")

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.RMSprop(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.RMSprop(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

        # max_grad_norm = MAX_GRAD_NORM 

        state = np.zeros((s_info,s_len))
        state = torch.from_numpy(state)
        bit_rate = DEFAULT_QUALITY
        default_vmaf = train_env[0].chunk_psnr[DEFAULT_QUALITY][0]
        last_quality = default_vmaf
        # action_vec = np.zeros(A_DIM)
        # action_vec[bit_rate] = 1

        time_stamp = 0

        episode_steps = 20
        # update_num = 1
        # batch_size = exploration_size * episode_steps #64
        gamma = 0.99
        gae_param = 0.95
        ent_coeff = 1
        # cl_coeff = 0.2
        memory = ReplayMemory(args.agent_num * episode_steps)
        state_ini = [state for _ in range(args.agent_num)]
        last_quality_ini = [last_quality for _ in range(args.agent_num)]
        # bit_rate_ini = [bit_rate for i in range(agent_num)]

        max_QoE = {}
        max_QoE[0] = -99999

        for epoch in range(int(args.epochT)):
            for agent in range(args.agent_num):
                states = []
                actions = []
                rewards_comparison = []
                rewards = []
                values = []
                returns = []
                advantages = []

                # get initial state and bitrate
                state = state_ini[agent]
                last_quality = last_quality_ini[agent]
                # bit_rate = bit_rate_ini[agent]

                for _ in range(episode_steps):

                    prob = model_actor(state.unsqueeze(0).type(dtype))
                    action = prob.multinomial(num_samples=1).detach()
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
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

                    # delay, sleep_time, buffer_size, rebuf, \
                    #     video_chunk_size, next_video_chunk_sizes, \
                    #     end_of_video, video_chunk_remain = \
                    #         env_abr[agent].get_video_chunk(bit_rate) ## sample in the environment of virtual player
                    delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, next_video_chunk_psnrs, \
                        end_of_video, video_chunk_remain, _, curr_chunk_psnrs \
                            = train_env[agent].get_video_chunk(bit_rate)
                    
                    time_stamp += delay  # in ms
                    time_stamp += sleep_time  # in ms

                    # reward is video quality - rebuffer penalty - smooth penalty
                    # -- lin scale reward --
                    # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    #         - REBUF_PENALTY * rebuf \
                    #         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                    #                                 VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                    # -- log scale reward --
                    curr_quality = curr_chunk_psnrs[bit_rate]
                    sm_dif_p = max(curr_quality - last_quality, 0)
                    sm_dif_n = max(last_quality - curr_quality, 0)
                    reward = quality_penalty * curr_quality \
                            - rebuffer_penalty * rebuf \
                                - smooth_penalty_p * sm_dif_p \
                                    - smooth_penalty_n * sm_dif_n \
                                        - 2.661618558192494

                    rew_ = float(max(reward, -4 * rebuffer_penalty)/20.)

                    rewards.append(rew_)
                    rewards_comparison.append(torch.tensor([reward]))

                    last_quality = curr_quality

                    # retrieve previous state
                    if end_of_video:
                        state = np.zeros((s_info,s_len))
                        state = torch.from_numpy(state)
                        last_quality = default_vmaf
                        break

                    # dequeue history record
                    state = np.roll(state, -1, axis=1)

                    # this should be S_INFO number of terms
                    state[0, -1] = float(last_quality / DB_NORM_FACTOR)  # last quality
                    state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
                    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
                    state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
                    state[6, :A_DIM] = np.array(next_video_chunk_psnrs) / DB_NORM_FACTOR

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

                # restore the initial state
                state_ini[agent] = state
                last_quality_ini[agent] = last_quality

                # one last step
                R = torch.zeros(1, 1)
                if end_of_video == False:
                    v = model_critic(state.unsqueeze(0).type(dtype))
                    v = v.detach().cpu()
                    R = v.data
                #=========================finish one episode=========================
                # compute returns and GAE(lambda) advantages:
                # values.append(Variable(R))
                # R = Variable(R)
                # A = Variable(torch.zeros(1, 1))
                # for i in reversed(range(len(rewards))):
                #     td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                #     A = float(td) + gamma * gae_param * A
                #     advantages.insert(0, A)
                #     # R = A + values[i]
                #     R = gamma * R + rewards[i]
                #     # R = Variable(R)
                #     returns.insert(0, R)

                ## compute returns and advantages with Monte Carlo sampling
                values.append(Variable(R))
                R = Variable(R)
                # td = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    R = gamma * R + rewards[i]
                    returns.insert(0, R)
                    
                    td = R - values[i]
                    advantages.insert(0, td)
                # store usefull info:
                # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                # memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
                if torch.eq(states[0][0], torch.from_numpy(np.zeros((s_info,s_len)))).sum() == s_info * s_len: ## judge if states[0] equals to torch.from_numpy(np.zeros((S_INFO,S_LEN)))
                    memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
                else:
                    memory.push([states, actions, returns, advantages])
        
            # policy grad updates:
            model_actor.zero_grad()
            model_critic.zero_grad()

            # mini_batch
            batch_size = memory.return_size()
            batch_states, batch_actions, batch_returns, batch_advantages = memory.sample_cuda(batch_size)
            # pdb.set_trace()
            probs_pre = model_actor(batch_states.type(dtype))
            values_pre = model_critic(batch_states.type(dtype))

            # actor_loss
            prob_value = torch.gather(probs_pre, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))
            policy_loss = -torch.mean(prob_value * batch_advantages.type(dtype))
            loss_ent = ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))
            actor_loss = policy_loss + loss_ent

            # critic_loss
            vf_loss = (values_pre - batch_returns.type(dtype)) ** 2 # V_\theta - Q'
            critic_loss = 0.5 * torch.mean(vf_loss)

            # update
            actor_total_loss = policy_loss + loss_ent
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_total_loss.backward()
            optimizer_actor.step()
            critic_loss.backward()
            optimizer_critic.step()

            ## test and save the model
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(policy_loss.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(critic_loss.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))

            if epoch % int(args.valid_i) == 0:
                mean_value = valid(args, valid_env, model_actor, epoch, test_log_file, save_folder)
                # entropy_weight = 0.95 * entropy_weight
                ent_coeff = 0.95 * ent_coeff
                # save(model_actor, model_critic, summary_dir)
                save_models_a2c(logging, save_folder, \
                                args.name, model_actor, \
                                model_critic, epoch, max_QoE, mean_value)
            
            clean_file_cache(log_file, log_file_name + '_record')

def main():
    train_a2c()

if __name__ == '__main__':
    main()