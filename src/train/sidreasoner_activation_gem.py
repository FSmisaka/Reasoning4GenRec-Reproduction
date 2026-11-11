"""
SIDReasoner Stage 2(GEM): 推理激活 SFT 的 GEM 损失变体

与 src/train/sidreasoner_activation.py 完全相同的数据/训练流程,
仅把 SFT 的 CE 损失换成 GEM(见 src/train/sidreasoner_gem.py)。
默认从 GEM Stage 1 的 final_checkpoint(runs/sidreasoner/
<category>_stage1_sft[_own]_gem)继续训练; 超参见
config/sidreasoner.py 的 training.activation.gem; 输出目录自动加
_gem 后缀。

启动(双卡示例):
  DATASET=games .venv/bin/python -m torch.distributed.run \
      --nproc_per_node 2 --master_port 29520 -m src.train.sidreasoner_activation_gem
或 scripts/sidreasoner/activation_gem.sh / make games-sidreasoner-activation-gem
"""

from src.train.sidreasoner_activation import main as stage2_activation_main


def main(cfg=None):
    stage2_activation_main(cfg, loss="gem")


if __name__ == "__main__":
    main()
