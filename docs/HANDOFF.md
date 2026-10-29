# 交接说明（HANDOFF）

写于 2026-10-29，第二次更新（第一次在 macOS 机器上，第二次在 RTX 5070 Ti 机器上跑完三组实验之后）。

用途：让另一个开发者在另一台机器上，不读聊天记录也能接着干。

## 这是什么

把 2025 年 1 月《人工智能基础》课程作业里的 FER2013 表情识别项目重建了一遍，让它满足三件事：

1. 每个写出来的数字都能被第三方重算；
2. 部署相关的说法（ONNX 导出、CPU 延时）有代码和测量支撑，不是叙述；
3. 后续实验（重训、官方划分、模型对比）有现成的入口，不需要重写脚本。

个人作品集用途，最终会挂到简历上，所以 README 里的每个数都必须经得起追问。

## 已经完成

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 复现课程作业的两个 checkpoint | 完成（在原机器上） | 第 5 轮 59.43% / 宏 F1 0.5354，第 10 轮 63.39% / 0.6046（3,589 张留出集） |
| 划分可复现 | 完成 | `src/split.py --mode legacy` 逐行复现当初的测试集；在 Windows 上用另一份 CSV 重跑，七类 support 逐个一致（498/52/545/881/588/414/611，合计 3,589） |
| ONNX 导出 + 数值校验 | 完成（在原机器上） | 单文件 43.7 MB，与 PyTorch 最大 logit 差 5.2e-06 |
| CPU 延时基准（PyTorch vs ONNX） | 完成（在原机器上） | 单图 1.71 → 1.07 ms（1.60×），batch 32 60.57 → 36.95 ms（1.64×） |
| **官方划分 40 轮** | **完成（本次）** | **63.81% / 宏 F1 0.6295**，7,178 张未参与训练的 PublicTest+PrivateTest，`results/metrics_official.json` |
| **随机划分 40 轮** | **完成（本次）** | **63.95% / 宏 F1 0.6310**，与 63.39% 同口径，`results/metrics_legacy40.json` |
| **6 模型对比表** | **完成（本次）** | 3 CNN + 3 MLP，20 轮 official，参数量/准确率/宏 F1/CPU 延时，见 README 和 `results/latency.json` |
| **训练曲线** | **完成（本次）** | `results/figures/curves_cnn40-official.png`、`curves_cnn40-legacy.png` |
| 中文注释与文档 | 完成 | 每个源文件开头有用途说明，README / `docs/experiment-log.md` 有完整记录 |

## 结论（写进 README 的三条）

1. **40 轮不比 10 轮强多少。** legacy 63.39% → 63.95%（+0.56 pp），验证曲线从第 8 轮左右就平了。训练损失掉到 0.02 而验证损失从 1.1 涨到 2.44，是过拟合，不是训练不够。
2. **两个划分的差距只有 0.14 pp**（63.95% legacy vs 63.81% official）。随机划分泄漏身份这件事在方法上依然成立，但对这个模型来说，泄漏带来的虚高没有想象中大——可以如实写，不必夸大。
3. **参数几乎免费，延时不是。** cnn-light（0.94 M）只比 emotioncnn（10.9 M）低 3 pp，却只用 1/6 的 CPU 时间和 1/11.6 的参数；MLP 快 5–60 倍、低 12–18 pp。

## 还没做（按优先级）

1. **把两个原始 checkpoint 放出去。** 它们被 `.gitignore` 排除，别人 clone 下来跑不了 `scripts/reproduce_published.sh`，63.39% 那一行就无法被第三方复算，这和「每个数字都能重算」的定位冲突。做法：GitHub Release 附件，或 Git LFS。**这是当前最大的复现缺口。**
2. **类别加权 / focal loss。** fear 和 sad 的 F1 都在 0.5 上下，是把宏 F1 拉下来的主因。
3. **数据增强（水平翻转 / 小角度仿射）。** 在 official 划分上评估。
4. 可选：把 `src/latency.py` 扩展成同时测 ONNX，让 6 模型表也能给出 ONNX 的毫秒数。

## 环境（本次实测通过）

Windows 11 + **RTX 5070 Ti**（16,303 MiB，驱动 610.88，compute capability 12.0），CPU 是 AMD Ryzen 7 9700X（8C/16T）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

实测版本：Python 3.13.7、torch 2.11.0+cu128、torchvision 0.26.0+cu128、CUDA 12.8、onnx 1.23.2、onnxruntime 1.31.1、scikit-learn 1.9.1、pandas 3.1.0、matplotlib 3.11.2、numpy 2.5.4。`torch.cuda.is_available()` 返回 True。

- **Blackwell 必须 cu128。** 装 cu121 会报 `no kernel image is available`。
- **下载源速度实测**：`download.pytorch.org` 6.9 MB/s（可用）；阿里云 `mirrors.aliyun.com/pytorch-wheels/cu128/` 只有 0.03 MB/s（别用，会以为卡死）；南大、清华镜像的该路径 404。
- 单卡 5070 Ti 上 emotioncnn 每轮约 2.6 s（batch 256，25,838 张），40 轮约 2 分钟；CPU 上原先是 81 秒/轮，差 30 倍以上。

## 数据与权重

两者都不在 Git 里（`.gitignore` 已排除）。

| 文件 | 大小 | sha256 |
| --- | --- | --- |
| `data/fer2013/fer2013.csv`（本次下载） | 301,072,768 B | `4af58868d510d40405c6f5fb12e63087d37c26ce9128a897cde0196e8a3e4205` |
| `data/fer2013/fer2013.csv`（原机器，Kaggle 原版） | 287 MB | 前 16 位 `3b8d9617d1017f34` |
| `checkpoints/model_epoch_10.pth` | 43.7 MB | 前 16 位 `a3709206d8a7b261`（**不在本机，也不需要**，只影响 reproduce_published） |

本次没走 Kaggle（无凭据），用的是 HuggingFace 镜像 `DerrickUnleashed/FER-2013` 的 `icml_face_data.csv`。两个文件的差异**只在格式，不在内容**：

- 表头不同：镜像是 `emotion, Usage, pixels`（逗号后带空格），Kaggle 原版是 `emotion,pixels,Usage`。因此 sha256 必然不同，不能拿 hash 当同一性证据。
- 行尾不同：镜像是 LF，Kaggle 原版是 CRLF。
- **同一性的硬证据**：35,887 行、Usage 分布 28,709 / 3,589 / 3,589；`--mode legacy` 重跑后测试集七类 support 与 `results/metrics_epoch10.json` 逐个相同。即发布数字用的就是这份数据。

代码现在对两种表头都兼容（`src/split.py`、`src/dataset.py` 在读入后 `df.columns = [c.strip() for c in df.columns]`），因为镜像的列名带前导空格会让 `df["Usage"]` 直接 KeyError。

## 常用命令（全部本次实跑过）

```bash
# 生成两种划分（第二个 --out 不能省，否则会覆盖 legacy 的文件）
python -m src.split --mode legacy
python -m src.split --mode official --out data/fer2013_official

# 训练（GPU）。注意：没有 --mode 参数，官方划分靠 --data-dir 指定
python -m src.train --model emotioncnn --data-dir data/fer2013_official \
  --device cuda --batch-size 256 --epochs 40 --threads 8 --tag cnn40-official

# 评估
python -m src.evaluate --checkpoint checkpoints/cnn40-official_best.pth \
  --split data/fer2013_official/test.csv \
  --out results/metrics_official.json --figure results/figures/cm_official.png

# 训练曲线（--baseline 默认 24.55 是 legacy 的多数类基线，official 用 24.71）
python -m src.plot_history results/history_cnn40-official.json --baseline 24.71 \
  --out results/figures/curves_cnn40-official.png

# 6 模型的 CPU 单图延时
python -m src.latency --pair cnn-light=checkpoints/m20-cnn-light_best.pth \
  --pair emotioncnn=checkpoints/m20-emotioncnn_best.pth --out results/latency.json

# 复现已发布的数字（需要两个原始 checkpoint，本机上跑不了）
./scripts/reproduce_published.sh
```

## 本次修掉的问题（都不在环境，在仓库里）

1. **`src/train.py` 每次跑完都崩在写 history。** `json.dumps` 直接吃 `vars(args)`，里面 `checkpoint_dir` / `out_dir` 是 `Path`，抛 `TypeError: Object of type WindowsPath is not JSON serializable`。结果是 checkpoint 存下来了、`history_*.json` 从来没写出来（所以之前的仓库里一个 history 都没有）。已改成把 `Path` 统一转 `str`。
2. **`NEXT_SESSION_PROMPT.md` 和旧 HANDOFF 的训练命令不成立。** 写的是 `--mode official`，但 `src/train.py` 没有 `--mode`，只有 `--data-dir`；照抄会直接报错。
3. **6 模型里没有 `cnn_deep`。** `src/model.py` 注册的是 `emotioncnn / cnn-light / cnn-small / mlp-small / mlp-medium / mlp-large`（中划线，不是下划线），`cnn_light` 这类名字也不对。做对比表时用的是真实存在的六个，`emotioncnn` 顶「深」这一格。
4. **`src/split.py` docstring 的划分行数少算了一行**（写 29,069 / 3,231 / 3,590，实际 29,068 / 3,230 / 3,589），已更正。
5. **CSV 表头带空格的兼容问题**（见上一节）。

## 已知的坑

1. **ONNX 默认外挂权重。** torch 2.6+ 的 `torch.onnx.export` 会把 >2GB 或按需把权重写成 `.onnx.data`。`src/export_onnx.py` 里已设 `external_data=False`，得到单文件 43.7 MB。别把这个参数删了。
2. **参数量口径。** `sum(p.numel())` 是 10,927,751；`state_dict` 里序列化的是 10,929,163 个标量，多出的 1,412 是 4 层 BatchNorm 的 `running_mean/running_var/num_batches_tracked`（buffer，不是可训练参数）。对外说 10.93M。
3. **随机划分会泄漏身份。** 同一张脸的不同帧可能落在训练集和测试集两侧。所以 63.39% / 63.95% 只能和自己比，不能和论文比。对外比较只用 `--mode official` 的 63.81%。
4. **`--mode legacy` 的划分是「复现历史」，不是「好划分」。** 保留它是为了让 2025 年的两个 checkpoint 有可复算的对照，新实验优先用 official。
5. **原课程代码里 `TrainThread.run()` 是 `pass`**（PyQt5 里的训练按钮没接上），`prepare_data.py` 里路径写死成 Windows 的 `C:/Users/...`。这份重建没有复用那两个脚本，只复用了模型结构和数据协议。
6. **旧 HANDOFF 写的 `torch 2.15.0` 是笔误**，torch 没有 2.15；README 里那一句已改成不写版本号（原机器的版本没记录下来，编一个反而破坏「数字可查」的定位）。

## 约定

- 新的实验，在 `docs/experiment-log.md` 里加一行：日期、配置、测试准确率、宏 F1、备注。
- 换 checkpoint 或换划分，`results/` 里**新增** JSON，不要覆盖 `metrics_epoch5.json` / `metrics_epoch10.json`。
- README 里的数字必须来自 `results/*.json`，不要手写。
- 别把 `data/`、`checkpoints/`、`.venv/` 提交进 Git。
