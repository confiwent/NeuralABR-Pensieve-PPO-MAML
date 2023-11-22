import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable

DEFAULT_QUALITY = 1
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
# DB_NORM_FACTOR = 100.0

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
S_INFO = 6
S_LEN = 8
A_DIM = 6
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY_LOG = 2.66  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_LIN = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class VirtualPlayer:
    def __init__(self, args, env, log_file):
        self.env = env
        self.args = args
        self.task_list = env.task_list

        ## get the information of virtual players (personality)
        # s_info, s_len, c_len, total_chunk_num, bitrate_versions, \
        #     quality_penalty, rebuffer_penalty, smooth_penalty_p, smooth_penalty_n \
        #         = env.get_env_info()
        # Video information
        self.s_info, self.s_len, self.total_chunk_num, self.quality_p, self.smooth_p = (
            S_INFO,
            S_LEN,
            CHUNK_TIL_VIDEO_END_CAP,
            1,
            SMOOTH_PENALTY,
        )
        self.bitrate_versions = VIDEO_BIT_RATE
        self.rebuff_p = REBUF_PENALTY_LIN if args.lin else REBUF_PENALTY_LOG
        self.br_dim = len(self.bitrate_versions)

        # QoE reward scaling
        self.scaling_lb = 4 * self.rebuff_p
        self.scaling_r = self.rebuff_p

        # define the state for rl agent
        self.state = np.zeros((self.s_info, self.s_len))

        # information of emulating the video playing
        self.last_bit_rate = DEFAULT_QUALITY
        self.time_stamp = 0.0
        self.end_flag = True
        self.video_chunk_remain = self.total_chunk_num

        # log files, recoding the video playing
        self.log_file = log_file

        # information of action mask
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.video_chunk_sizes = env.get_video_size()

    def step(self, bit_rate):
        # execute a step forward
        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self.env.get_video_chunk(bit_rate)

        # compute and record the reward of current chunk
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        self.video_chunk_remain = video_chunk_remain

        # compute reward of Quality of experience
        if self.args.lin:
            # -- lin scale reward --
            reward = (
                self.bitrate_versions[bit_rate] / M_IN_K
                - self.rebuff_p * rebuf
                - self.smooth_p
                * np.abs(
                    self.bitrate_versions[bit_rate]
                    - self.bitrate_versions[self.last_bit_rate]
                )
                / M_IN_K
            )
        else:
            # -- log scale reward --
            log_bit_rate = np.log(
                self.bitrate_versions[bit_rate] / float(self.bitrate_versions[0])
            )
            log_last_bit_rate = np.log(
                self.bitrate_versions[self.last_bit_rate]
                / float(self.bitrate_versions[0])
            )
            reward = (
                log_bit_rate
                - self.rebuff_p * rebuf
                - self.smooth_p * np.abs(log_bit_rate - log_last_bit_rate)
            )
        rew_ = float(max(min(reward, self.rebuff_p), self.scaling_lb) / self.scaling_r)
        # reward_norm = self.reward_filter(rew_)
        reward_norm = rew_

        self.last_bit_rate = bit_rate

        # -------------- logging -----------------
        # log time_stamp, bit_rate, buffer_size, reward
        self.log_file.write(
            str(self.time_stamp)
            + "\t"
            + str(self.bitrate_versions[bit_rate])
            + "\t"
            +
            # str(np.sum(self.action_mask)) + '\t' +
            str(buffer_size)
            + "\t"
            + str(rebuf)
            + "\t"
            + str(video_chunk_size)
            + "\t"
            + str(delay)
            + "\t"
            + str(reward)
            + "\n"
        )
        self.log_file.flush()

        ## dequeue history record
        self.state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        self.state[0, -1] = self.bitrate_versions[bit_rate] / float(
            np.max(self.bitrate_versions)
        )  # last quality
        self.state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
        self.state[2, -1] = (
            float(video_chunk_size) / float(delay) / M_IN_K
        )  # kilo byte / ms
        self.state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, : self.br_dim] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        self.state[5, -1] = np.minimum(
            video_chunk_remain, self.total_chunk_num
        ) / float(self.total_chunk_num)

        state_ = np.array([self.state])
        state_ = torch.from_numpy(state_).type(dtype)

        self.end_flag = end_of_video
        if self.end_flag:
            self.reset_play()
        return state_, reward_norm, end_of_video

    def set_task(self, idx):
        self.env.set_task(idx)

    def reset_play(self):
        self.state = np.zeros((self.s_info, self.s_len))

        self.last_bit_rate = DEFAULT_QUALITY
        self.video_chunk_remain = self.total_chunk_num
        self.time_stamp = 0.0

        self.past_bandwidth_ests = []
        self.past_errors = []

        self.log_file.write("\n")
        self.log_file.flush()

    def clean_file_cache(self, file_name, max_file_size=4.096e7):
        file_size = os.stat(file_name).st_size
        if file_size > max_file_size:
            self.log_file.seek(0)
            self.log_file.truncate()
