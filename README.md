# PPO and A2C based adaptive bitrate algorithms (Variants of Pensieve, Pytorch version)

Re-implementation of existing neural ABR algorithms with Pytorch, same paradigm as Pensieve.

## User guide

To train the policy with a __linear or logarithmic__ video quality metric, refer to ```./main.py```

To train the policy with a __perceptual__ video quality metric, i.e., VMAF, refer to ```./variant_vmaf/main_vmaf.py```

> In these files, you can run the Pensive training by ```python main.py --a2c``` or ```python ./variant_vmaf/main_vmaf.py --a2c```. Please refer to ```./script/train.sh``` for more details.

- We also have implemented two variants of Pensieve: Pensieve with A3C algorithm (a well-established DRL method), and Pensieve with MAML algorithm (a meta-reinforcement learning method). You can run their training processes by ```python ./variant_vmaf/main_vmaf.py --a2c``` and ```python ./variant_vmaf/main_vmaf.py --a2br```, respectively.

implemented by pytorch and trained using GPU

>Note that the original version of Pensieve using asynchronous advantage actor-critic algorithm (A3C) to train the policy, which can only implementated on CPU. Our A2C version removes the asynchronous setting and use GPU to accelerate the speed of NNs training. 

Further improvements are ongoing...
