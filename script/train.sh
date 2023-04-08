echo "Train the ABR policy by A2C (Pensieve) with vmaf QoE metric!"

CUDA_VISIBLE_DEVICES=0 python ./variant_vmaf/main_vmaf.py --a2c --name rl --agent-num 10