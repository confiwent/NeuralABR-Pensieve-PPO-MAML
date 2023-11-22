# echo "Train the ABR policy by A2BR with Lin/Log QoE metric!"

echo "Train the ABR policy by A2C with lin/log QoE metric!"
echo "python main.py --maml --valid-i 1000 --proba"

CUDA_VISIBLE_DEVICES=0 python main.py --maml --valid-i 1000 --proba