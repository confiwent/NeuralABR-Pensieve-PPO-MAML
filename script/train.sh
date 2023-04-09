echo "Train the ABR policy by A2BR with vmaf QoE metric!"

CUDA_VISIBLE_DEVICES=0 python ./variant_vmaf/main_vmaf.py --a2br --name maml