# echo "Train the ABR policy by A2BR with vmaf QoE metric!"

# CUDA_VISIBLE_DEVICES=1 python ./variant_vmaf/main_vmaf.py --a2br --name maml --valid-i 1000 --proba --init --ent-coeff 0.1

echo "Train the ABR policy by A2C with vmaf QoE metric!"

CUDA_VISIBLE_DEVICES=1 python ./variant_vmaf/main_vmaf.py --a2c --valid-i 1000 --proba --name a2c --init