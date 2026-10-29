# 给下一个会话的提示词

把下面整段复制给新会话的 AI。它假设仓库已经 clone 到本地，且你在仓库根目录。

上一版这份文件写于三组实验**之前**，命令是错的（`src/train.py` 没有 `--mode`，只有 `--data-dir`），
`cnn_deep` 这个模型也从未存在过。三组实验已经在 RTX 5070 Ti 上跑完并写进 README，
所以这一版换成「接下来做什么」。

---

我在 `fer2013-emotion-cnn` 这个仓库里继续一个 FER2013 表情识别的个人项目。这个仓库的定位是
**每个写出来的数字都能被第三方重算**，所以任何改动都不能让某个数字失去出处。

**第一步，先读，不要写代码。**
读 `docs/HANDOFF.md`（状态、缺口、已知的坑）、`README.md`（已发布的结果）、
`docs/experiment-log.md`（每行一次运行 + 三条结论）、`src/train.py` 和 `src/split.py` 的参数。
读完用不超过十行向我复述：哪些数字已经验证过、接下来的三个任务是什么、有哪些不能碰的文件。

**第二步，确认环境。**
`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。
如果 GPU 没认到，先停下来报告，不要自行换 CUDA 版本硬装——Blackwell 需要 cu128 的 wheel，
其他版本会报 `no kernel image is available`。

**第三步，按优先级做三件事。**

1. **放出两个原始 checkpoint（最高优先级，这是唯一的复现缺口）。**
   `checkpoints/model_epoch_5.pth` 和 `model_epoch_10.pth` 被 `.gitignore` 排除，所以
   `scripts/reproduce_published.sh` 和 README 里 63.39% 那一行，别人 clone 下来跑不了。
   先问我权重文件在哪个盘/哪台机器，再决定用 GitHub Release 附件还是 Git LFS。
   做完之后 `README.md` 的「Known limitations」第 1 条要改写。

2. **类别不平衡。** fear 和 sad 的 F1 都在 0.5 上下，是宏 F1 的主要拖累。
   加类别加权（或 focal loss）作为 `src/train.py` 的一个 flag，默认关闭以保持已有结果不变；
   在 **official 划分** 上跑 20 轮对比：基线的 cnn-light 是 60.43% / 0.5937，
   emotioncnn 是 63.00% / 0.6239（见 README 的模型对比表）。
   重点看 fear 和 sad 的 per-class recall 有没有动，而不是只看总准确率。

3. **数据增强。** 水平翻转 + 小角度仿射，同样作为 flag、默认关闭，同样在 official 划分上对比。
   注意 FER2013 是表情识别，水平翻转要确认标签语义不变（happy/sad 等对称，但左右脸遮挡类的样本可能不适合）。

**约束：**

- 不要改动或删除 `results/metrics_epoch5.json`、`results/metrics_epoch10.json` 和两个原始 checkpoint，它们是对照基准。
- 不要覆盖已有的 `results/metrics_*.json`，新实验用新文件名。
- 每完成一组，在 `docs/experiment-log.md` 追加一行（日期、配置、准确率、宏 F1、备注），并把新结论写进它的「Findings」段。
- README 里的数字只能来自 `results/*.json`，不要手写估计值。
- 显存不够就把 `--batch-size` 降到 128，不要改模型结构。
- 每次跑之前先告诉我要跑什么、大概多久；不要同时启动多个训练抢 GPU。
- 实验开关默认关闭：`--seed 42` + 原参数重跑必须能复现已发布的数字。

**最后，产出这份东西给我：**

```
任务 1（权重）：方案 = ______，我需要在哪台机器上做什么 = ______
任务 2（类别不平衡）：official 20 epochs
  baseline cnn-light    accuracy = __.__%  macro-F1 = 0.___  fear F1 = 0.___  sad F1 = 0.___
  weighted  cnn-light   accuracy = __.__%  macro-F1 = 0.___  fear F1 = 0.___  sad F1 = 0.___
任务 3（数据增强）：同上，两组
结论：值不值得写进简历的一句话 = ______
```

如果某一步失败，报告原始报错文本，不要自己改架构或降级指标来「让它跑通」。
