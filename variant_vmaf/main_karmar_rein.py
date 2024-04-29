import numpy as np
import torch
import os
import torch.nn.functional as F

import argparse
import random
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer

# from decision_transformer.d4rl import get_dataset
from qoe_to_go import QoE_predictor_model
import matplotlib.pyplot as plt
from utils.data_loader import get_trajs


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = "cpu"
    return device


def experiment(variant):

    device_id = variant["device"]
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    q2go_model = QoE_predictor_model().to(device)

    model_checkpoint_path = variant["q2go_cpt"]
    q2go_model.load_state_dict(torch.load(model_checkpoint_path))
    q2go_model.eval()

    dataset_path = variant["traj_path"]
    trajectories = get_trajs(dataset_path, q2go_model, device)

    # save all path information into separate lists
    states = np.concatenate(trajectories["observations"], axis=0)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    state_dim = states.shape[1]
    act_dim = trajectories["actions"][0].shape[1]

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]

    def get_batch(batch_size=512, max_len=K):

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for _ in range(batch_size):
            traj_idx = random.randint(0, len(trajectories["observations"]) - 1)
            si = random.randint(
                0, len(trajectories["observations"][traj_idx]) - max_len
            )

            # get sequences from dataset
            s.append(
                trajectories["observations"][traj_idx][si : si + max_len].reshape(
                    1, -1, state_dim
                )
            )
            a.append(
                trajectories["actions"][traj_idx][si : si + max_len].reshape(
                    1, -1, act_dim
                )
            )
            r.append(
                trajectories["rewards"][traj_idx][si : si + max_len].reshape(1, -1, 1)
            )
            d.append(
                trajectories["terminals"][traj_idx][si : si + max_len].reshape(1, -1)
            )
            rtg.append(
                trajectories["returns"][traj_idx][si : si + max_len].reshape(1, -1, 1)
            )

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))

            # timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            # rtg.append(
            #     discount_cumsum(trajectories["rewards"][traj_idx][si:], gamma=0.99)[
            #         : s[-1].shape[1] + 1
            #     ].reshape(1, -1, 1)
            # )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * 1, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    scale = 160
    env_targets = [3600, 1800]
    max_ep_len = 512
    mode = variant.get("mode", "normal")

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():

                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )

                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        action_tanh=False,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
    )

    model = model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )

    warmup_steps = variant["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: F.cross_entropy(a_hat, a.long()),
        # eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    loss_list = []
    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        loss_list.append(outputs["training/train_loss_mean"])

    return loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=1e5)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--num_steps_per_iter", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--traj_path", type=str, default="./traces_dataset/oracle8_trajs-6000.pkl"
    )
    parser.add_argument(
        "--q2go_cpt",
        type=str,
        default="./checkpoints/q2go/Q2GO_predictor.pt",
    )

    args = parser.parse_args()

    loss_list = experiment(variant=vars(args))

    plt.plot(loss_list)
    plt.savefig("loss_exp.png")
