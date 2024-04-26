import os
import  numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = 'create_data/results_lin/'

class QoE_predictor_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.QoE_predictor_model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.optimizer_inv = torch.optim.Adam(self.QoE_predictor_model.parameters(), lr=1e-4)
    
    def forward(self,x):

        y = self.QoE_predictor_model(x)

        return y
        
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

        check_list = ['expert']
        #check_list = ['mpc']
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

def get_batch(dataset,start_ptr):
    batch_size = len(dataset['observations'])
    discounts_all = 0.99 ** np.arange(100)
    obs = np.zeros((batch_size,4))
    rtg = np.zeros((batch_size,1))

    for i in range(batch_size):
        obs[i,:] = [0.15276, 0.122, dataset['observations'][i][start_ptr,1], dataset['observations'][i][start_ptr,3]]
        rewards = dataset['rewards'][i][start_ptr+1:]
        discounts = discounts_all[:len(rewards)]
        rtg[i,:] = ((discounts * rewards).sum())/160
    
    return obs,rtg

def main():
    dataset = get_dataset(env)
    model = QoE_predictor_model()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_ptr = 0
    epoch = 0
    loss_list = []
    for epoch in range(165):
        for step in range(45):
            obs,rtg = get_batch(dataset,step)

            QoE_pred = model(torch.tensor(obs).to(device,dtype=torch.float))
            loss = F.mse_loss(QoE_pred,torch.tensor(rtg).to(device,dtype=torch.float))
            loss.backward()

            model.optimizer_inv.step()
            model.optimizer_inv.zero_grad()
            

        print(f'epoch ={epoch} ')
        print(f'{loss:8.6f} \n')
        loss_list.append(loss.detach().item())
            

    torch.save(model.state_dict(), 'weights/checkpoint/QoE_predictor.pt')

    return loss_list

if __name__ == '__main__':
    loss_list = main()
    plt.plot(loss_list)
    plt.savefig('loss.png')