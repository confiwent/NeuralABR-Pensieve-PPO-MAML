import numpy as np
import torch, os

from utils.data_loader import get_throughput_char, get_qoe2go_estimation
from utils.helper import results_analysis

M_IN_K = 1000.
S_DIM = 16
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # kbps
TOTAL_CHUNK_NUM = 49
QUALITY_PENALTY = 0.8469011 #dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887

def evaluate_episode_abr(test_env, dt_model, q2go_model, log_path_ini, device, s_dim=S_DIM, log_dir='./results_valid/'):
    DEFAULT_QUALITY = 1
    test_env.set_env_info(0, 0, 0, TOTAL_CHUNK_NUM, VIDEO_BIT_RATE, \
                            QUALITY_PENALTY, REBUF_PENALTY, SMOOTH_PENALTY_P, SMOOTH_PENALTY_N)
    bit_rate = DEFAULT_QUALITY
    last_bit_rate = DEFAULT_QUALITY
    last_quality = test_env.chunk_psnr[DEFAULT_QUALITY][0]
    state_dim = s_dim
    a_dim = len(VIDEO_BIT_RATE)
    state = np.zeros((state_dim))
    action_vec = np.zeros(a_dim)
    action_vec[bit_rate] = 1
    s_info, s_len, c_len, total_chunk_num, bitrate_versions, \
        quality_penalty, rebuffer_penalty, smooth_penalty_p, smooth_penalty_n \
            = test_env.get_env_info()
    a_dim = len(bitrate_versions)

    # initialize recording files
    all_file_name = test_env.all_file_names
    log_path = log_path_ini + '_' + all_file_name[test_env.trace_idx]
    log_file = open(log_path, 'w')

    # start to download video
    for video_count in range(len(all_file_name)):
        '''for a network trace '''
        past_throughputs = np.zeros((1, 4))
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        # states = torch.zeros((0, state_dim), device=device, dtype=torch.float32)
        actions = torch.from_numpy(action_vec).reshape(1, a_dim).to(device=device, dtype=torch.float32)
        # actions = torch.zeros((0, a_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        returns_est = torch.zeros(1, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.zeros(1, device=device, dtype=torch.long).reshape(1, 1)
        time_stamp = 0
        chunk_id = 0
        while True:
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, next_video_chunk_psnrs, \
                    end_of_video, video_chunk_remain, _, curr_chunk_psnrs \
                        = test_env.get_video_chunk(bit_rate)
            
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smooth penalty
            curr_quality = curr_chunk_psnrs[bit_rate]
            sm_dif_p = max(curr_quality - last_quality, 0)
            sm_dif_n = max(last_quality - curr_quality, 0)
            reward = quality_penalty * curr_quality \
                        - rebuffer_penalty * rebuf \
                            - smooth_penalty_p * sm_dif_p \
                                - smooth_penalty_n * sm_dif_n \
                                    - 2.661618558192494

            last_bit_rate = bit_rate
            last_quality = curr_quality

            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(bitrate_versions[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
            log_file.flush()

            # ========== get action and reward ============
            # add padding
            actions = torch.cat([actions, torch.zeros((1, a_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
            # ========== get state ============
            BUFFER_NORM_FACTOR = 10.
            state[0] = video_chunk_size / delay / M_IN_K  # kilo byte / ms # throughput
            state[1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec # buffer size
            # last quality
            # state[2] = parse[1] / float(np.max(ACTION_SELECTED))
            state[2] = delay / M_IN_K  # chunk download time
            state[3] = np.minimum(video_chunk_remain, total_chunk_num) / float(
                total_chunk_num
            )  # fraction of remaining chunks
            state[4 : 4 + a_dim] = (
                np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            )  # next chunk sizes
            state[4 + a_dim : 4 + 2 * a_dim] = (
                np.array(next_video_chunk_psnrs) / 100
            )  # vmaf values [0, 100] # next chunk vmaf values

            cur_state = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = torch.cat([states, cur_state], dim=0)

            # ========== get Qoe2Go prediction ============
            # get observation of QoE2Go
            throughput_current = video_chunk_size / delay / M_IN_K  # kilo byte / ms
            throughput_mean, throughput_std = get_throughput_char(
                past_throughputs, throughput_current
            )
            past_throughputs = np.roll(past_throughputs, -1, axis=1)
            past_throughputs[0, -1] = throughput_current
            buffer_size = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
            remain_chunks_num = np.minimum(video_chunk_remain, total_chunk_num) / float(
                total_chunk_num
            )

            # get QoE2Go estimation
            q2go_obs = [throughput_mean, throughput_std, buffer_size, remain_chunks_num]
            q2go_estimation = get_qoe2go_estimation(q2go_obs, q2go_model, device)
            cur_return = torch.from_numpy(np.array([q2go_estimation])).reshape(1,1).to(device=device, dtype=torch.float32)
            returns_est = torch.cat([returns_est, cur_return], dim=0)

            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (chunk_id+1)], dim=1)


            # ========== get action prediction ============
            # get action inference
            with torch.no_grad():
                prob = dt_model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns_est.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            bit_rate = int(torch.argmax(prob).squeeze().cpu().numpy())

            # set action
            action_vec = np.zeros(a_dim)
            action_vec[bit_rate] = 1
            action_vec = torch.from_numpy(action_vec).to(device=device, dtype=torch.float32)
            actions[-1] = action_vec
            rewards[-1] = reward

            chunk_id += 1

            # end of video  
            if end_of_video:
                last_quality = test_env.chunk_psnr[DEFAULT_QUALITY][0]
                state = np.zeros((state_dim))
                action_vec = np.zeros(a_dim)
                action_vec[DEFAULT_QUALITY] = 1
                log_file.write('\n')
                log_file.close()
                time_stamp = 0

                if video_count + 1 >= len(all_file_name):
                    break
                else:
                    log_path = log_path_ini + '_' + all_file_name[test_env.trace_idx]
                    log_file = open(log_path, 'w')
                    break

    # ============ results analysis ==============
    _, _, qoe_mean, qoe_std, _, _ = results_analysis(log_dir)
    return qoe_mean, qoe_std


def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros(
        (0, act_dim), device=device, dtype=torch.float32
    )  # empty tensor
    rewards = torch.zeros(0, device=device, dtype=torch.float32)  # empty tensor

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat(
            [actions, torch.zeros((1, act_dim), device=device)], dim=0
        )  # each step we add a zero padding
        rewards = torch.cat(
            [rewards, torch.zeros(1, device=device)]
        )  # reward is not used

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length
