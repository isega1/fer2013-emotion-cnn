# 交接说明（HANDOFF）

写于 2026-10-29。这份文件的目的：让另一个开发者在另一台机器上，不读聊天记录也能接着干。

## 这是什么

把 2025 年 1 月《人工智能基础》课程作业里的 FER2013 表情识别项目重建了一遍，让它满足三件事：

1. 每个写出来的数字都能被第三方重算；
2. 部署相关的说法（ONNX 导出、CPU 延时）有代码和测量支撑，不是叙述；
3. 后续实验（重训、官方划分、模型对比）有现成的入口，不需要重写脚本。

个人作品集用途，最终会挂到简历上，所以 README 里的每个数都必须经得起追问。

## 已经完成

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 复现课程作业的两个 checkpoint | 完成 | 第 5 轮 59.43% / 宏 F1 0.5354，第 10 轮 63.39% / 0.6046（3,589 张留出集），见 README |
| 划分可复现 | 完成 | `src/split.py --mode legacy` 逐行复现当初的测试集（3,589 行、顺序一致，仅 LF/CRLF 差异） |
| ONNX 导出 + 数值校验 | 完成 | 单文件 43.7 MB，与 PyTorch 最大 logit 差 5.2e-06 |
| CPU 延时基准 | 完成 | 单图 1.71 → 1.07 ms（1.60×），batch 32 60.57 → 36.95 ms（1.64×） |
| 中文注释与文档 | 完成 | 每个源文件开头有用途说明，README / `docs/experiment-log.md` 有完整记录 |

## 还没做（按优先级）

1. **官方划分重训。** `--mode official` 用 FER2013 原始的 PublicTest，数字才能和公开结果并排。当前 63.39% 是在随机划分上测的，同一张脸可能在训练集和测试集里都出现，只能内部比较。
2. **40 轮重训。** 第 5 轮到第 10 轮涨了 4 个点，明显没收敛。目标是看 40 轮 + 余弦退火能到多少。
3. **6 模型对比表。** `src/model.py` 里已经写好 `cnn_light`、`cnn_small`、`cnn_deep`、`mlp_small`、`mlp_medium`、`mlp_large`，`--model` 直接切换，缺的只是一张结果表。
4. 可选：类别加权 / focal loss（disgust 只有 52 张，F1 0.545；sad F1 0.471）。

## 环境

原机器是 macOS（Apple Silicon，CPU 训练 81 秒/轮，太慢）。目标机器是 **RTX 5070 Ti**。

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Blackwell 必须 cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 必须是 True。装了 cu121 甚至更旧的 wheel 会报 `no kernel image is available`——这是最容易浪费半小时的坑。

Python 3.14 已验证可用（torch 2.15.0 / onnxruntime 1.31.1 / scikit-learn 1.9.1 / pandas 3.1.0）。

## 数据与权重

两者都不在 Git 里（`.gitignore` 已排除）。

| 文件 | 大小 | sha256 前 16 位 |
| --- | --- | --- |
| `data/fer2013/fer2013.csv` | 287 MB | `3b8d9617d1017f34` |
| `checkpoints/model_epoch_10.pth` | 43.7 MB | `a3709206d8a7b261` |

- 数据来自 Kaggle 的 *Challenges in Representation Learning: Facial Expression Recognition Challenge*（ICML 2013 workshop），共 35,887 张 48×48 灰度图、7 类。
- 最快的搬法是从原机器整目录拷贝（data 574 MB + checkpoints 209 MB）；否则从 Kaggle 重新下载 `fer2013.csv`，再用 `python -m src.split --mode legacy|official` 生成划分。
- 权重是课程作业的原产物，别再训练覆盖它们（文件名固定，`src/train.py` 只会写 `checkpoints/<tag>_*.pth`）。

## 常用命令

```bash
# 复现已发布的数字（不需要训练）
./scripts/reproduce_published.sh

# 训练（GPU）
python -m src.train --mode official --device cuda --batch-size 256 --epochs 40 --threads 8 --tag cnn40-official

# 评估一个 checkpoint
python -m src.evaluate --checkpoint checkpoints/cnn40-official_best.pth \
  --split data/fer2013_official/test.csv \
  --out results/metrics_official.json --figure results/figures/cm_official.png

# 画训练曲线
python -m src.plot_history results/history_cnn40-official.json
```

## 已知的坑

1. **ONNX 默认外挂权重。** torch 2.6+ 的 `torch.onnx.export` 会把 >2GB 或按需把权重写成 `.onnx.data`。`src/export_onnx.py` 里已设 `external_data=False`，得到单文件 43.7 MB。别把这个参数删了。
2. **参数量口径。** `sum(p.numel())` 是 10,927,751；`state_dict` 里序列化的是 10,929,163 个标量，多出的 1,412 是 4 层 BatchNorm 的 `running_mean/running_var/num_batches_tracked`（buffer，不是可训练参数）。对外说 10.93M。
3. **随机划分会泄漏身份。** 同一张脸的不同帧可能落在训练集和测试集两侧。所以 63.39% 只能和自己比，不能和论文比。要对外比较只用 `--mode official`。
4. **`--mode legacy` 的划分是"复现历史"，不是"好划分"。** 保留它是为了让 2025 年的两个 checkpoint 有可复算的对照，新实验优先用 official。
5. **原课程代码里 `TrainThread.run()` 是 `pass`**（PyQt5 里的训练按钮没接上），`prepare_data.py` 里路径写死成 Windows 的 `C:/Users/...`。这份重建没有复用那两个脚本，只复用了模型结构和数据协议。

## 约定

- 新的实验，在 `docs/experiment-log.md` 里加一行：日期、配置、测试准确率、宏 F1、备注。
- 换 checkpoint 或换划分，`results/` 里**新增** JSON，不要覆盖 `metrics_epoch5.json` / `metrics_epoch10.json`。
- README 里的数字必须来自 `results/*.json`，不要手写。
- 别把 `data/`、`checkpoints/`、`.venv/` 提交进 Git。
