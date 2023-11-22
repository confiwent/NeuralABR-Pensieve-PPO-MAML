"""
In this file, maml_ppo algorithm is adopted to fine-tune the policy of rate adaptation, gae advantage function and multi-step return are used to calculate the gradients.

Add the reward normalization, using vmaf quality metric

Designed by kannw_1230@sjtu.edu.cn

"""

import os
import numpy as np
import random

import torch, time
import torch.optim as optim
from torch.autograd import Variable
import logging
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from maml_ppo import MAMLPPO
from test_maml_torch import valid
import env
import load_trace
from env_wrapper import VirtualPlayer
from variant_vmaf.utils.helper import check_folder

RANDOM_SEED = 28
DEFAULT_QUALITY = int(1)  # default video quality without agent

SUMMARY_DIR = "./Results/sim"

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train_maml_ppo(args):
    add_str = "maml"
    summary_dir = os.path.join(*[SUMMARY_DIR, add_str])
    check_folder(summary_dir)
    log_str = "lin" if args.lin else "log"
    summary_dir = os.path.join(*[summary_dir, log_str])
    check_folder(summary_dir)
    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    log_file_path = os.path.join(*[summary_dir, ts])
    check_folder(log_file_path)
    log_file_name = log_file_path + "/log"
    writer = SummaryWriter(log_file_path)
    mean_value = 0

    # define the parameters of ABR environments
    # _, _, _, _, bitrate_versions, _, _, _, _ = train_env.get_env_info()
    br_dim = 6
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()
    train_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        a2br=True,
    )

    torch.manual_seed(RANDOM_SEED)

    with open(log_file_name + "_record", "w") as log_file, open(
        log_file_name + "_test", "w"
    ) as test_log_file:
        agent = MAMLPPO(args, br_dim)
        if args.init:
            init_ckpt_path = os.path.join(
                *[summary_dir, "init_ckpt"]
            )  # Notice: ensure the correct model!
            agent.load(init_ckpt_path)

        steps_in_episode = args.ro_len

        vp_env = VirtualPlayer(args, train_env, log_file)
        task_num = len(vp_env.env.task_list)

        # while True:
        for epoch in range(int(1e5)):
            # agent.model_eval()
            # vp_env.reset_reward()

            # ---------- limit the file size -----------
            vp_env.clean_file_cache(log_file_name + "_record")

            iteration_replays = []
            iteration_policies = []

            for _ in range(task_num):
                clone = deepcopy(agent.actor)
                vp_env.env.set_task(_)
                task_replay = []

                # Fast Adapt
                for _ in range(args.adapt_steps):
                    train_episodes = agent.collect_steps(
                        clone, vp_env, n_episodes=steps_in_episode
                    )
                    _, _, clone = agent.fast_adapt(
                        clone, train_episodes, first_order=True
                    )
                    task_replay.append(train_episodes)

                # Compute Validation Loss
                valid_episodes = agent.collect_steps(
                    clone, vp_env, n_episodes=steps_in_episode
                )
                task_replay.append(valid_episodes)
                iteration_replays.append(task_replay)
                iteration_policies.append(clone)

            # training the models
            policy_loss_, entropy_loss_ = agent.meta_optimize(
                iteration_replays, iteration_policies
            )

            writer.add_scalar("Avg_Policy_loss", policy_loss_, epoch)
            writer.add_scalar("Avg_Entropy_loss", entropy_loss_, epoch)

            if epoch % int(args.valid_i) == 0 and epoch > 0:
                model_actor = agent.actor
                mean_value = valid(
                    args, model_actor, epoch, test_log_file, log_file_path
                )

                save_folder = os.path.join(*[log_file_path, "checkpoints"])
                check_folder(save_folder)
                agent.save(save_folder, epoch)
            writer.add_scalar("Avg_Return", mean_value, epoch)
            writer.flush()

            if epoch % int(100) == 0 and epoch > 0:
                agent.ent_coeff_decay()

        writer.close()
