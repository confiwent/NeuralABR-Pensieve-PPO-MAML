import os, pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


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
        self.optimizer_inv = torch.optim.Adam(
            self.QoE_predictor_model.parameters(), lr=1e-4
        )

    def forward(self, x):

        y = self.QoE_predictor_model(x)

        return y


def dataloader(
    batch_size, test_split, file_path="./traces_dataset/oracle8_trajs-3000.pkl"
):
    # load list from pkl files where the path is file_path
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)

    test_size = int(test_split * len(dataset))
    train_list, test_list = dataset[test_size:], dataset[:test_size]

    return traj_to_dataloader(batch_size, train_list), traj_to_dataloader(
        batch_size, test_list
    )


def traj_to_dataloader(batch_size, data_list):

    train_rtg_tensor, train_obs_tensor = get_data_tensor(data_list)

    train_ds = TensorDataset(train_obs_tensor, train_rtg_tensor)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_dl


def get_data_tensor(dataset):

    keys_list = ["observations", "actions", "rewards", "terminals"]
    BUFFER_NORM_FACTOR = 10.0
    M_IN_K = 1000.0
    total_chunk_num = 48
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for log_file in dataset:

        observations = []
        action = []
        rewards = []
        terminals = []
        past_throughputs = np.zeros((1, 5))

        for _, line in enumerate(log_file):

            parse = line.split()
            parse = [float(x) for x in parse]

            if len(parse) < 2:
                break
            # action
            action.append(int(parse[1]))
            throughput_current = parse[4] / parse[5] / M_IN_K  # kilo byte / ms
            throughput_mean, throughput_std = get_throughput_char(
                past_throughputs, throughput_current
            )
            buffer_size = parse[2] / BUFFER_NORM_FACTOR  # 10 sec
            remain_chunks_num = np.minimum(parse[6], total_chunk_num) / float(
                total_chunk_num
            )

            observations.append(
                [throughput_mean, throughput_std, buffer_size, remain_chunks_num]
            )

            rewards.append(parse[-1])

        # without start & end
        action = action[1:]
        observations = observations[:-1]
        rewards = rewards[:-1]

        terminals = [0] * len(rewards)
        terminals[-1] = 1

        all_actions.append(np.array(action))
        all_observations.append(np.array(observations))
        all_rewards.append(np.array(rewards))
        all_terminals.append(np.array(terminals))

    dataset_traj = {}
    for key in keys_list:
        dataset_traj[key] = locals()["all_" + key]

    train_obs = []
    train_rtg = []

    for start_ptr in range(total_chunk_num - 2):
        obs, rtg = get_training_data(dataset_traj, start_ptr)
        train_obs.extend(obs)
        train_rtg.extend(rtg)

    train_obs_np = np.asarray(train_obs)
    train_rtg_np = np.asarray(train_rtg)
    del dataset_traj
    del train_rtg
    del train_obs

    train_rtg_tensor = torch.tensor(train_rtg_np, dtype=torch.float32)
    train_obs_tensor = torch.tensor(train_obs_np, dtype=torch.float32)

    del train_obs_np
    del train_rtg_np

    return train_rtg_tensor, train_obs_tensor


def get_throughput_char(past_bandwidths, new_bandwidth):
    """
    Calculates the mean and standard deviation of past bandwidths with a new bandwidth.

    Args:
        past_bandwidths (numpy.ndarray): Array of past bandwidth values.
        new_bandwidth (float): New bandwidth value to be added to the past bandwidths.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the past bandwidths.

    """
    past_bandwidths_ = np.roll(past_bandwidths, -1, axis=1)[0].tolist()
    while past_bandwidths_[0] == 0.0:
        past_bandwidths_ = past_bandwidths_[1:]
    past_bandwidths_[-1] = new_bandwidth

    bw_mean = np.mean(past_bandwidths_)
    bw_std = np.std(past_bandwidths_)
    return bw_mean, bw_std


def get_training_data(dataset, start_ptr):
    batch_size = len(dataset["observations"])
    discounts_all = 0.99 ** np.arange(100)
    obs = []
    rtg = []

    for i in range(batch_size):
        obs.append(dataset["observations"][i][start_ptr, :4])
        rewards = dataset["rewards"][i][start_ptr + 1 :]
        discounts = discounts_all[: len(rewards)]
        rtg.append(((discounts * rewards).sum()) / 1000)

    return obs, rtg


class Trainer(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.batch_size = 1000
        self.test_split = 0.2
        self.train_dl, self.test_dl = dataloader(
            self.batch_size,
            self.test_split,
            self.file_path,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = QoE_predictor_model().to(self.device)

    def train(self, max_epochs=100):
        loss_list = []
        for epoch in range(max_epochs):
            for obs, rtg in self.train_dl:
                QoE_pred = self.model(obs)
                loss = F.mse_loss(QoE_pred, rtg)
                loss.backward()

                self.model.optimizer_inv.step()
                self.model.optimizer_inv.zero_grad()

            print(f"epoch ={epoch} ")
            print(f"{loss:8.6f} \n")
            loss_list.append(loss.detach().item())

        torch.save(self.model.state_dict(), "weights/checkpoint/QoE_predictor.pt")

        return loss_list


def main():
    file_path = "./traces_dataset/oracle8_trajs-3000.pkl"
    trainer = Trainer(file_path)
    loss_list = trainer.train()
    plt.plot(loss_list)
    plt.savefig("loss.png")


if __name__ == "__main__":
    main()
