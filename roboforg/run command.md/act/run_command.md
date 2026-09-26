# act 评估代码

conda activate gym_hil
cd ~/project/RoboForge

python -m roboforg.workflows.act_1.eval_act \
  --checkpoint="checkpoints/act/pick__act__20260924_232506/epoch_0200.ckpt" \
  --show-viewer=true \
  --num-episodes=10 \
  --print-policy-output=true



# act 训练代码

conda activate gym_hil
cd ~/project/RoboForge

bash roboforg/workflows/act_1/train_act.sh