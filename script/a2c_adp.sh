# echo "Train the ABR policy by A2BR with vmaf QoE metric!"

# CUDA_VISIBLE_DEVICES=1 python ./variant_vmaf/main_vmaf.py --a2br --name maml --valid-i 1000 --proba --init --ent-coeff 0.1

echo "Adaptation for A2BR by A2C with vmaf QoE metric!"

CUDA_VISIBLE_DEVICES=1 python ./variant_vmaf/main_vmaf.py --a2c --valid-i 20 --name adpa2c --init --adp --epochT 1000 --Ada2c --tr-folder puffer_adp_0210