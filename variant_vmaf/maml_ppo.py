import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import logging
import learn2learn as l2l
from torch import autograd
from replay_memory import ReplayMemory
from model_ac_vmaf import Actor, Critic

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class MAMLPPO():
    def __init__(self, a_dim,
                 adapt_lr=1e-3, meta_lr=1e-4, 
                 adapt_steps=3, ppo_steps=5,
                 gamma=0.99, tau=0.95,
                 policy_clip=0.2,
                 seed=42,
                 device=None, name="MAMLPPO", tensorboard_log="./logs"):
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.a_dim = a_dim

        self.gamma = gamma
        self.tau = tau
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.policy_clip = policy_clip
        self.ppo_steps = ppo_steps

        # ---- initialize models ----
        self.actor = Actor(a_dim).type(dtype)
        self.critic = Critic(a_dim).type(dtype)

        # ---- set optimizer for actor and critic
        self.optimizer = torch.optim.Adam(self.actor.parameters(), meta_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), adapt_lr)

    def save(self, path="./"):
        torch.save(self.critic.state_dict(), path + "/critic.pt")
        torch.save(self.actor.state_dict(), path + "/actor.pt")


    def load(self, path="./"):
        self.critic.load_state_dict(torch.load(path + "/baseline.pt"))
        self.actor.load_state_dict(torch.load(path + "/policy.pt"))

    def compute_adv(self, done, value, values, rewards):
        'Calculates the advantages and returns for a trajectories.'
        gamma, gae_param = self.gamma, self.tau
        advantages = []
        returns = []

        # ==================== finish one episode ===================
        # one last step
        R = torch.zeros(1, 1)
        if done == False:
            v = value.cpu()
            R = v.data

        values.append(Variable(R).type(dtype))
        R = Variable(R).type(dtype)
        A = Variable(torch.zeros(1, 1)).type(dtype)

        rewards_ = np.array(rewards)
        rewards_ = torch.from_numpy(rewards_).type(dtype)

        for i in reversed(range(len(rewards))):
            td = rewards_[i].data + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
            A = td + gamma * gae_param * A
            advantages.insert(0, A)
            # R = A + values[i]
            R = gamma * R + rewards_[i].data
            returns.insert(0, R)

        return advantages, returns

    def collect_steps(self, actor, env, n_episodes):
        env.env.reset()
        done = True
        explo_bit_rate = 1
        states = []
        actions = []
        rewards = []
        values = []
        memory = ReplayMemory(500)

        for _ in range(n_episodes):
            # record the current state, observation and action
            if not done:
                states.append(state_)
                actions.append(action)
                values.append(value)

            bit_rate = explo_bit_rate

            state_, reward_norm, done = env.step(bit_rate)
            rewards.append(reward_norm)

            # value, action = actor.explore(ob_, state_, action_mask_)
            with torch.no_grad():
                prob = actor.forward(state_)
                value = self.critic(state_)
                action = prob.multinomial(num_samples=1)
            # value, action = agent.explore(ob_, state_)
            explo_bit_rate = int(action.squeeze().cpu().numpy())
            if done:
                explo_bit_rate = 1
                break

        # compute returns and GAE(lambda) advantages:
        if len(states) != len(rewards):
            if len(states) + 1 == len(rewards):
                rewards = rewards[1:]
            else:
                print('error in length of states!')
        advantages, returns = self.compute_adv(done, value, values, rewards)
        replay = [states, actions, returns, advantages]
        memory.push(replay)

        # ----- update critic ----
        batch_states, _, batch_returns, _ = memory.sample_cuda(memory.return_size())
        v_pre = self.critic(batch_states)
        # value loss
        vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
        loss_value = 0.5 * torch.mean(vfloss1)
        loss_critic = loss_value
        self.optimizer_critic.zero_grad()
        # loss_actor.backward(retain_graph=False)
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm = 3., norm_type = 2)     
        self.optimizer_critic.step()

        del memory
        return replay
    
    def dual_ppo_loss(self, train_episodes, old_policy, new_policy):
        memory = ReplayMemory(500)
        memory.push(train_episodes)
        batch_states, batch_actions, _, batch_advantages = memory.sample_cuda(memory.return_size())
        # old_prob
        probs_old = old_policy(batch_states).detach()
        prob_value_old = torch.gather(probs_old, dim=1, \
                                        index=batch_actions.type(dlongtype)).detach()
        # new prob
        probs = new_policy(batch_states)
        prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))

        # ratio
        ratio = prob_value / (1e-6 + prob_value_old)

        # clip loss
        surr1 = ratio * batch_advantages.type(dtype)  # surrogate from conservative policy iteration
        surr2 = ratio.clamp(1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages.type(dtype)
        loss_clip_ = torch.min(surr1, surr2)
        loss_clip_dual = torch.where(torch.lt(batch_advantages.type(dtype), 0.), \
                                        torch.max(loss_clip_, 1.5 * batch_advantages.type(dtype)), \
                                            loss_clip_)
        loss_clip_actor = -torch.mean(loss_clip_dual)

        # entropy loss
        ent_latent = 0.1 * torch.mean(probs * torch.log(probs + 1e-6))
        del memory
        return loss_clip_actor + ent_latent

    def maml_a2c_loss(self, memory, actor):
        # obtain policy loss
        batch_states, batch_actions, _, batch_advantages = memory.sample_cuda(memory.return_size())
        probs = actor(batch_states)
        prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))
        loss = -torch.mean(prob_value * batch_advantages.type(dtype))
        ent = 0.1 * torch.mean(probs * torch.log(probs + 1e-5))
        return loss + ent

    def fast_adapt(self, clone, train_episodes, first_order=False):
        memory = ReplayMemory(500)
        memory.push(train_episodes)
        second_order = not first_order
        loss = self.maml_a2c_loss(memory, clone)
        gradients = autograd.grad(loss,
                                clone.parameters(),
                                retain_graph=second_order,
                                create_graph=second_order)
        
        del memory
        return l2l.algorithms.maml.maml_update(clone, self.adapt_lr, gradients)
        

    def meta_loss(self, iteration_replays, iteration_policies, policy):
        mean_loss = 0.0
        for _ in range(len(iteration_replays)):
            task_replays = iteration_replays[_]
            old_policy = iteration_policies[_]
            train_replays = task_replays[:-1]
            valid_episodes = task_replays[-1]
            new_policy = l2l.clone_module(policy)

            # Fast Adapt
            for _ in range(len(train_replays)):
                train_episodes = train_replays[_]
                new_policy = self.fast_adapt(new_policy, train_episodes, first_order=False)

            # Compute Surrogate Loss
            surr_loss = self.dual_ppo_loss(valid_episodes, old_policy, new_policy)
            mean_loss += surr_loss
        mean_loss /= len(iteration_replays)
        return mean_loss

    def meta_optimize(self, iteration_replays, iteration_policies):
        for ppo_epoch in range(self.ppo_steps):
            loss = self.meta_loss(iteration_replays, iteration_policies, self.actor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss