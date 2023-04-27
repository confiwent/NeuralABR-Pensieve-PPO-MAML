import numpy as np
import random

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
# BITRATE_LEVELS = 6
# TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
SAMPLE_NUM = 26

def importance_sampling(all_cooked_bw):
    mean_value = [np.mean(i) for i in all_cooked_bw]

    value, base = np.histogram(mean_value, bins= SAMPLE_NUM)

    sampling_list = []
    for n in range(1, len(base)):
        member = []
        for idx in range(len(mean_value)):
            if mean_value[idx] > base[n-1] and mean_value[idx] <= base[n]:
                member.append(idx)
            if mean_value[idx] == base[n-1]:
                member.append(idx)
        sampling_list.append(member)

    return sampling_list

def initialize_tasks(task_list, all_file_names):
    task2idx = {}
    for task_id in task_list:
        for trace_id in range(len(all_file_names)):
            if task_id in all_file_names[trace_id]:
                try:
                    task2idx[task_id].append(trace_id)
                except:
                    task2idx[task_id] = []
                    task2idx[task_id].append(trace_id)
    assert(len(task2idx)==len(task_list))
    return task2idx

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names, video_size_file, video_psnr_file, random_seed=RANDOM_SEED, a2br = False):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.task_list = ['bus.ljansbakken', 'car.snaroya', 'ferry.nesoddtangen', 'metro.kalbakken', 'norway_bus', 'norway_car', 'norway_metro', 'norway_train', 'norway_tram', 'amazon', 'yahoo', 'facebook', 'youtube']

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.all_file_names = all_file_names
        self.sampling_list = importance_sampling(all_cooked_bw)
        self.task2idx = initialize_tasks(self.task_list, all_file_names)
        self.task_id = int(0)
        self.a2br_flag = a2br

        self.video_chunk_counter = 0
        self.buffer_size = 0
        
        self.s_info = 17
        self.s_len = 10
        self.c_len = 3
        self.bitrate_version = [300, 750, 1200, 1850, 2850, 4300]
        self.br_dim = len(self.bitrate_version)
        self.qual_p = 0.85
        self.rebuff_p = 28.8
        self.smooth_p = 1
        self.smooth_n = 0.3

        # pick a trace file
        # self.idx_location = 0
        # idx_member = self.sampling_list[self.idx_location]
        # idx_str = np.random.randint(len(idx_member))
        # self.trace_idx = idx_member[idx_str]
        if self.a2br_flag:
            idx_member = self.task2idx[self.task_list[self.task_id]]
            idx_str = np.random.randint(len(idx_member))
            self.trace_idx = idx_member[idx_str]
        else:
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.br_dim):
            self.video_size[bitrate] = []
            with open(video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.chunk_psnr = {} # video quality of chunks
        for bitrate in range(self.br_dim):
            self.chunk_psnr[bitrate] = []
            with open(video_psnr_file + str(bitrate)) as f:
                for line in f:
                    self.chunk_psnr[bitrate].append((float(line.split()[0])))

        self.total_chunk_num = len(self.video_size[0])
        self.chunk_length_max = self.total_chunk_num

    def reset(self):
        if self.a2br_flag:
            idx_member = self.task2idx[self.task_list[self.task_id]]
            idx_str = np.random.randint(len(idx_member))
            self.trace_idx = idx_member[idx_str]
        else:
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_chunk_counter = 0
        self.buffer_size = 0
    
    def set_task(self, idx):
        self.task_id = int(idx)
        idx_member = self.task2idx[self.task_list[self.task_id]]
        idx_str = np.random.randint(len(idx_member))
        self.trace_idx = idx_member[idx_str]
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        return self.task_id == len(self.task_list) - 1

    def reset_task(self):
        self.task_id = int(0)

    # pythran export set_env_info(int, int, int, int, int list, float, float)
    def set_env_info(self, s_info, s_len, c_len, chunk_num, br_version, qual_p, rebuff_p, smooth_p, smooth_n):
        self.s_info = s_info
        self.s_len = s_len
        self.c_len = c_len
        self.total_chunk_num = chunk_num
        self.chunk_length_max = chunk_num
        self.bitrate_version = br_version
        self.br_dim = len(self.bitrate_version)
        self.qual_p = qual_p
        self.rebuff_p = rebuff_p
        self.smooth_p = smooth_p
        self.smooth_n = smooth_n

    # pythran export get_env_info(None)
    def get_env_info(self):
        return self.s_info, self.s_len , self.c_len, self.total_chunk_num, self.bitrate_version, self.qual_p, self.rebuff_p, self.smooth_p, self.smooth_n

    def get_video_size(self):
        return self.video_size

    def get_video_psnr(self):
        return self.chunk_psnr

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.br_dim

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        curr_chunk_sizes = []
        curr_chunk_psnrs = []
        for i in range(self.br_dim):
            curr_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            curr_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunk_num - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_chunk_num:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            # self.total_chunk_num = random.randint(10, int(self.chunk_length_max))
            
            if self.a2br_flag:
                idx_member = self.task2idx[self.task_list[self.task_id]]
                idx_str = np.random.randint(len(idx_member))
                self.trace_idx = idx_member[idx_str]
            else:
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
                if self.trace_idx >= len(self.all_cooked_time):
                    self.trace_idx = 0  
            # 
            # ===== importance sampling =====    
            # if self.idx_location + 1 >= SAMPLE_NUM:
            #     self.idx_location = 0
            # else:
            #     self.idx_location += 1
            # while len(self.sampling_list[self.idx_location]) == 0:
            #     if self.idx_location + 1 >= SAMPLE_NUM:
            #         self.idx_location = 0
            #     else:
            #         self.idx_location += 1
            # idx_member = self.sampling_list[self.idx_location]
            # idx_ptr = np.random.randint(len(idx_member))
            # self.trace_idx = idx_member[idx_ptr]

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        next_video_chunk_psnrs = []
        for i in range(self.br_dim):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            next_video_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            next_video_chunk_psnrs, \
            end_of_video, \
            video_chunk_remain, \
            curr_chunk_sizes, \
            curr_chunk_psnrs
