"""
SIDReasoner Stage 1(GEM): 增强式 SID-语言对齐 SFT 的 GEM 损失变体

与 src/train/sidreasoner_sft.py 完全相同的数据/训练流程, 仅把
SFT 的 CE 损失换成 GEM(ICLR 2025, arXiv:2408.16673, 官方实现
https://github.com/liziniu/GEM 的 sft_trainer_v2.py 移植)。
超参(beta/h/print_entropy)见 config/sidreasoner.py 的
training.sft.gem; 输出目录自动加 _gem 后缀, 与 CE 基线互不覆盖;
评估/早停仍用 CE eval_loss, 与基线可比。

启动(双卡示例):
  DATASET=games .venv/bin/python -m torch.distributed.run \
      --nproc_per_node 2 --master_port 12341 -m src.train.sidreasoner_sft_gem
或 scripts/sidreasoner/sft_gem.sh / make games-sidreasoner-sft-gem
"""

from src.train.sidreasoner_sft import main as stage1_sft_main


def main(cfg=None):
    stage1_sft_main(cfg, loss="gem")


if __name__ == "__main__":
    main()
