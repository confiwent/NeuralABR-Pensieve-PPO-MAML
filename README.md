# PPO and A2C based adaptive bitrate algorithm (Variants of Pensieve)

PG methods for learning an neural ABR policy, same paradigm as Pensieve.

To train the policy with the linear or logarithmic video quality metric, refer to ```./main.py```

To train the policy with perceptual video quality metric, i.e., VMAF, refer to ```./variant_vmaf/main_vmaf.py``` and ```./script/train.sh```

implemented by pytorch and trained using GPU

Note that the original version of Pensieve using asynchronous advantage actor-critic algorithm (A3C) to train the policy, which can only implementated on CPU. Our A2C version removes the asynchronous setting and use GPU to accelerate the speed of NNs training. 

Further improvements are ongoing...
