import os
import collections
import numpy as np
import random

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#with suppress_output():
    ## d4rl prints out a variety of warnings
    #import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    trace = 'HSDPA'
    keys_list = ['observations', 'actions', 'rewards', 'terminals']
    ACTION_SELECTED = [300,750,1200,1850,2850,4300]
    BUFFER_NORM_FACTOR = 10.0
    M_IN_K = 1000.0
    total_chunk_num = 48
    dataset = {}
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    cooked_files = os.listdir(env + trace + '/')

    #trace_files = os.listdir('create_data/trace_'+ trace + '/cooked_test_traces/')[40:]

    for cooked_file in cooked_files:
        file_path = env + trace + '/' + cooked_file
        
        # if 'expert' not in file_path and 'mpc' not in file_path:
        #     continue

        #check_list = ['expert']
        check_list = ['mpc']
        if all(item not in file_path for item in check_list):
            continue
        
        observations = []
        action = []
        rewards = []
        terminals = []

        with open(file_path, 'r') as f:
            """
                str(time_stamp / M_IN_K) + '\t' +
                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                str(buffer_size) + '\t' +
                str(rebuf) + '\t' +
                str(video_chunk_size) + '\t' +
                str(delay) + '\t' +
                str(video_chunk_remain) + '\t' + 
                str(next_video_chunk_sizes[0]) + '\t' +
                str(next_video_chunk_sizes[1]) + '\t' +
                str(next_video_chunk_sizes[2]) + '\t' +
                str(next_video_chunk_sizes[3]) + '\t' +
                str(next_video_chunk_sizes[4]) + '\t' +
                str(next_video_chunk_sizes[5]) + '\t' +
                str(reward) + '\n')
            """

            for i, line in enumerate(f):

                parse = line.split()
                parse = [float(x) for x in parse]
                
                if len(parse) < 2 :
                    break               
                #action
                action.append([int(parse[1] == bitrate) for bitrate in ACTION_SELECTED])
                
                state = np.zeros((11))

                state[0] = parse[4] / parse[5] / M_IN_K # kilo byte / ms
                state[1] = float(parse[2] / BUFFER_NORM_FACTOR)  # 10 sec
                # last quality
                #state[2] = parse[1] / float(np.max(ACTION_SELECTED))  
                state[2] = parse[5] / M_IN_K / BUFFER_NORM_FACTOR 
                state[3] = np.minimum(parse[6], total_chunk_num) / float(total_chunk_num)
                state[4 : 4+len(ACTION_SELECTED)] = np.array(parse[7:13]) / M_IN_K / M_IN_K # mega byte
                            
                observations.append(state)
                
                rewards.append(parse[-1])

                #rewards.append(int( parse[2] <30 and parse[2] > 20))
            
            #without start & end
            action = action[1:]
            observations = observations[:-1]
            rewards = rewards[:-1]
            
            terminals = [0] * len(rewards)
            terminals[-1] = 1
            
        all_actions.append(np.array(action))
        all_observations.append(np.array(observations))
        # all_returns = all_returns + returns
        all_rewards.append(np.array(rewards))
        all_terminals.append(np.array(terminals))

    
    # all_actions = np.array(all_actions)
    # all_observations = np.array(all_observations)
    # # all_returns = np.array(all_returns)
    # all_rewards = np.array(all_rewards) 
    # all_terminals = np.array(all_terminals)

    for key in keys_list:
        dataset[key] = locals()['all_' + key]

    return dataset

def sequence_dataset(env, max_path_length):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env) 
    #dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0] #203040
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        #if done_bool or final_timestep:
        if done_bool:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            # if 'maze2d' in env.name:
            #     episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
