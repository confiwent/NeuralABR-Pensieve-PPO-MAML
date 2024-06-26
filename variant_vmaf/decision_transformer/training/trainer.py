import numpy as np
import torch

import time, os


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        scheduler=None,
        ema=None,
        ts=None,
        eval_fns=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ema = ema
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.qoe_list = [-9999]
        self.traj_idx = 0
        self.start_time = time.time()
        self.current_valid_q_mean = -9999
        self.current_valid_q_std = -9999
        self.ts = ts
        if os.path.exists(f"./checkpoints/dt_{self.ts}") is False:
            os.mkdir(f"./checkpoints/dt_{self.ts}")

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.ema is not None:
                self.ema.update(self.model.parameters())
        if self.scheduler is not None:
            self.scheduler.step()

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        # ----------- start the evaluation -------------
        if iter_num % 10 == 0 or iter_num == 1:
            eval_start = time.time()
            self.model.eval()
            if self.ema is not None:
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model.parameters())
            # for eval_fn in self.eval_fns:
            output_mean, output_std = self.eval_fns(self.model)
            self.current_valid_q_mean = output_mean
            self.current_valid_q_std = output_std
            if output_mean >= max(self.qoe_list):
                torch.save(
                    self.model.state_dict(),
                    f"./checkpoints/dt_{self.ts}/dt_model_{iter_num}_r{output_mean}.pt",
                )
                self.qoe_list.append(output_mean)
            #     for k, v in outputs.items():
            #         logs[f'evaluation/{k}'] = v
            if self.ema is not None:
                self.ema.restore(self.model.parameters())

            logs["time/total"] = time.time() - self.start_time
            logs["time/evaluation"] = time.time() - eval_start
            logs["evaluation/valid_QoE_mean"] = self.current_valid_q_mean
            logs["evaluation/valid_QoE_std"] = self.current_valid_q_mean

            # for k in self.diagnostics:
            #     logs[k] = self.diagnostics[k]

            if print_logs:
                print("=" * 80)
                print(f"Iteration {iter_num}")
                for k, v in logs.items():
                    print(f"{k}: {v}")
        else:
            logs["evaluation/valid_QoE_mean"] = self.current_valid_q_mean
            logs["evaluation/valid_QoE_std"] = self.current_valid_q_mean

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size, self.traj_idx
        )
        self.traj_idx = self.traj_idx + 1
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
