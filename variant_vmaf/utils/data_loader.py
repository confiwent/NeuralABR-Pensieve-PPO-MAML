import os
import collections, torch
import numpy as np
import random, pickle


def get_throughput_char(past_bandwidths, new_bandwidth):
    """
    Calculates the mean and standard deviation of past bandwidths with a new bandwidth.

    Args:
        past_bandwidths (numpy.ndarray): Array of past bandwidth values.
        new_bandwidth (float): New bandwidth value to be added to the past bandwidths.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the past bandwidths.

    """
    past_bandwidths_ = np.roll(past_bandwidths, -1, axis=1)[0, :]
    # print(past_bandwidths_)
    # print(past_bandwidths)
    while past_bandwidths_[0] == 0.0 and len(past_bandwidths_) > 1:
        past_bandwidths_ = past_bandwidths_[1:]
    past_bandwidths_[-1] = new_bandwidth

    bw_mean = np.mean(past_bandwidths_)
    bw_std = np.std(past_bandwidths_)
    return bw_mean, bw_std


def get_qoe2go_estimation(obs, q2go_model, device="cpu"):
    obs_np = np.asarray(obs)
    del obs

    obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(device)
    del obs_np
    with torch.no_grad():
        QoE_pred = q2go_model(obs_tensor)
    return QoE_pred.cpu().numpy()[0]


def get_trajs(data_path, q2go_model, device="cpu"):

    keys_list = ["observations", "actions", "rewards", "terminals", "returns"]
    ACTION_SELECTED = [300, 750, 1200, 1850, 2850, 4300]
    a_dim = len(ACTION_SELECTED)
    BUFFER_NORM_FACTOR = 10.0
    M_IN_K = 1000.0
    total_chunk_num = 48
    dataset = {}
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_returns = []

    with open(data_path, "rb") as f:
        traj_dataset = pickle.load(f)

    for log_file in traj_dataset:

        observations = []
        action = []
        rewards = []
        terminals = []
        returns = []
        past_throughputs = np.zeros((1, 4))

        for _, line in enumerate(log_file):

            parse = line.split()
            parse = [float(x) for x in parse]

            if len(parse) < 2:
                break
            # action
            action_vec = np.zeros(a_dim)
            action_vec[int(parse[1])] = 1
            action.append(action_vec)
            throughput_current = parse[4] / parse[5] / M_IN_K  # kilo byte / ms
            throughput_mean, throughput_std = get_throughput_char(
                past_throughputs, throughput_current
            )
            past_throughputs = np.roll(past_throughputs, -1, axis=1)
            past_throughputs[0, -1] = throughput_current
            buffer_size = parse[2] / BUFFER_NORM_FACTOR  # 10 sec
            remain_chunks_num = np.minimum(parse[6], total_chunk_num) / float(
                total_chunk_num
            )

            q2go_obs = [throughput_mean, throughput_std, buffer_size, remain_chunks_num]
            q2go_estimation = get_qoe2go_estimation(q2go_obs, q2go_model, device)

            state = np.zeros((16))

            state[0] = parse[4] / parse[5] / M_IN_K  # kilo byte / ms
            state[1] = float(parse[2] / BUFFER_NORM_FACTOR)  # 10 sec
            # last quality
            # state[2] = parse[1] / float(np.max(ACTION_SELECTED))
            state[2] = parse[5] / M_IN_K
            state[3] = np.minimum(parse[6], total_chunk_num) / float(total_chunk_num)
            state[4 : 4 + a_dim] = np.array(parse[7:13]) / M_IN_K / M_IN_K  # mega byte
            state[4 + a_dim : 4 + 2 * a_dim] = (
                np.array(parse[13:19]) / 100
            )  # vmaf values [0, 100]

            observations.append(state)
            rewards.append(parse[-1])
            returns.append(q2go_estimation)

        # without start & end
        action = action[1:]
        observations = observations[:-1]
        rewards = rewards[:-1]
        returns = returns[:-1]
        terminals = [0] * len(rewards)
        terminals[-1] = 1

        all_actions.append(np.array(action))
        all_observations.append(np.array(observations))
        # all_returns = all_returns + returns
        all_rewards.append(np.array(rewards))
        all_terminals.append(np.array(terminals))
        all_returns.append(np.array(returns))

    for key in keys_list:
        dataset[key] = locals()["all_" + key]

    return dataset
