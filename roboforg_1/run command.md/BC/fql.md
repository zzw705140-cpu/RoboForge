# 训练
conda activate gym_hil
cd /home/xinyu/project/RoboForge

新训练：
bash 'roboforg_1/workflows/BC/bc+fql/train_fql.sh'
继续训练：
bash 'roboforg_1/workflows/BC/bc+fql/train_fql.sh' \
  --resume-checkpoint='/实际路径/epoch_0100.ckpt' \
  --num-epochs=1000


# 评估
conda activate gym_hil
cd /home/xinyu/project/RoboForge
bash 'roboforg_1/workflows/BC/bc+fql/eval_fql.sh' \
  --checkpoint='checkpoints/fql/pick__fql__20261017_164959/epoch_0100.ckpt' \
  --show-viewer=true \
  --num-episodes=20 \
  --print-policy-output=false

# GPU监控指令

watch -n 1 nvidia-smi