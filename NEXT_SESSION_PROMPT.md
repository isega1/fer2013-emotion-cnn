# 给另一台机器上的 AI 的提示词

把下面整段复制过去。它假设仓库已经 clone 到本地，且你在仓库根目录。

---

我在 `fer2013-emotion-cnn` 这个仓库里继续一个 FER2013 表情识别的个人项目。这台机器是 RTX 5070 Ti。

请按这个顺序做：

**第一步，先读，不要写代码。**
读 `docs/HANDOFF.md`（状态、缺口、已知的坑）、`README.md`（已发布的结果）、`docs/experiment-log.md`（实验记录）、`src/train.py` 和 `src/split.py` 的参数。读完用不超过十行向我复述：哪些数字已经验证过、接下来要产出的三个数字是什么、有哪些不能碰的文件。

**第二步，确认环境。**
运行 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。
如果 GPU 没认到，先停下来报告，不要自行换 CUDA 版本硬装——Blackwell 需要 cu128 的 wheel，其他版本会报 `no kernel image is available`。

**第三步，按 HANDOFF 的优先级跑三组实验，跑完一组报一组。**

1. 官方划分、40 轮：
   `python -m src.train --mode official --device cuda --batch-size 256 --epochs 40 --threads 8 --tag cnn40-official`
   然后 `python -m src.evaluate --checkpoint checkpoints/cnn40-official_best.pth --split data/fer2013_official/test.csv --out results/metrics_official.json --figure results/figures/cm_official.png`
2. 随机划分、40 轮（和现有的 63.39% 同口径）：
   `--mode legacy --tag cnn40-legacy`，评估用 `data/fer2013/test.csv`。
3. 6 模型对比：`--model` 依次取 `cnn_light / cnn_small / cnn_deep / mlp_small / mlp_medium / mlp_large`，每个 20 轮、official 划分，结果汇总成一张表（参数量、准确率、宏 F1、单图 CPU 延时）。

**约束：**

- 不要改动或删除 `results/metrics_epoch5.json`、`results/metrics_epoch10.json` 和两个原始 checkpoint，它们是对照基准。
- 每完成一组，在 `docs/experiment-log.md` 追加一行（日期、配置、准确率、宏 F1、备注）。
- README 里的数字只能来自 `results/*.json`，不要手写估计值。
- 显存不够就把 `--batch-size` 降到 128，不要改模型结构。
- 每次跑之前先告诉我你要跑什么、大概多久；不要一次启动多个训练抢 GPU。

**最后，产出这份东西给我（我要贴回另一个对话里更新简历）：**

```
official split, 40 epochs:
  accuracy = __.__%   macro-F1 = 0.___   best epoch = __
legacy split, 40 epochs:
  accuracy = __.__%   macro-F1 = 0.___   best epoch = __
6-model table: (粘贴表格)
剩余瓶颈：____________________
```

如果某一步失败，报告原始报错文本，不要自己改架构或降级指标来"让它跑通"。
