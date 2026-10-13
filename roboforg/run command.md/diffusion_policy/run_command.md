# Diffusion Policy 评估

conda activate gym_hil
cd ~/project/RoboForge

python -m roboforg.workflows.BC.diffusion_policy.eval_diffusion_policy \
  --checkpoint="checkpoints/diffusion_policy/pick__diffusion_policy__20261001_004838/epoch_0200.ckpt" \
  --show-viewer=true \
  --num-episodes=10 \
  --print-policy-output=false


# Diffusion Policy 训练

conda activate gym_hil
cd ~/project/RoboForge

bash roboforg/workflows/BC/diffusion_policy/train_diffusion_policy.sh


# GPU监控指令

watch -n 1 nvidia-smi