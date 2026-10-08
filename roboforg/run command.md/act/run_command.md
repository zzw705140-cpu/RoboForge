# act 评估代码

conda activate gym_hil
cd ~/project/RoboForge

python -m roboforg.workflows.act_1.eval_act \
  --checkpoint="checkpoints/act/pick__act__20260925_115405/epoch_0700.ckpt" \
  --show-viewer=true \
  --num-episodes=10 \
  --print-policy-output=false



# act 训练代码

conda activate gym_hil
cd ~/project/RoboForge

bash roboforg/workflows/act_1/train_act.sh