# AICOMP D-FINE-X RGB baseline 恢复工程

恢复两组独立的 RGB-only baseline。旧配置和划分已丢失，本工程是新的可复现实验，
不能保证与原服务器上的结果完全一致。RGB-only 是用户指定的基线，完整赛题要求三模态。

## 环境

```bash
source /home/fbohan/miniconda3/etc/profile.d/conda.sh
conda activate AICOMP
cd /home/fbohan/AIC
```

环境位于 `/home/fbohan/miniconda3/envs/AICOMP`，Python 3.11，PyTorch 2.5.1、
torchvision 0.20.1（CUDA 12.4）。安装完成后将保存 `requirements-lock.txt`。
官方源码：<https://github.com/Peterande/D-FINE>，提交
`956d1709314c2c6a4df6f34de232054578a7449f`。
预训练：官方 `dfine_x_obj2coco.pth`，仅使用权重，不下载额外训练数据。
环境级 `LD_LIBRARY_PATH` 已隔离系统 CUDA 12.2 的旧库，防止其覆盖 wheel 自带的 CUDA 12.4。
应先激活环境再手动运行 Python；启动器已自行设置正确路径。

## 实验

| 实验 | 训练 | 验证 | GPU | epochs |
|---|---:|---:|---|---:|
| rgb1600 | 1600 | 独立 400 | 0,1,2,3 | 100 |
| rgb2000 | 2000 | 无独立验证集 | 4,5,6,7 | 100 |

共同配置：D-FINE-X/HGNetv2-B5，12 类，总 batch 32（每卡 8），基础尺寸 640，
训练随机多尺度约 480–800；AdamW，主学习率 2.5e-4、骨干学习率 2.5e-6，
weight decay 1.25e-4，3 epochs 线性预热后逐步余弦下降至各组峰值的 1%。
AMP、TF32、SyncBN、EMA(decay=.9999, warmups=1000)、梯度范数裁剪 .1，
保留官方检测器 denoising、辅助损失和 GO-LSD。

增强使用官方 RandomPhotometricDistort、RandomZoomOut、RandomIoUCrop、
RandomHorizontalFlip、Resize 和边界框清理。第 91–100 轮关闭颜色扰动、
ZoomOut、IoUCrop 和多尺度，保留翻转、Resize。不使用多模型融合。

使用官方模型、损失和训练单步，但采用 `scripts/train_baseline.py` 的训练循环：
不在增强切换后回滚优化器或 epoch，完整训练 100 轮；全量版不构建验证数据加载器。
12 类分类头重新初始化，避免错误套用 COCO/Objects365 类别映射。

## 数据与划分

仅解压官方训练 ZIP 的 RGB 与标签；不读取测试集训练，不人工修改测试结果。
解压时检查 ZIP CRC，逐图解码，检查标签、重复文件名和划分间完全相同图像。
归一化标签转换为 COCO 格式，类别仍为 0–11；超出图像边界的框裁剪到图像内并计数。
固定 seed=20260929，按每图类别存在情况做多标签分层，精确 1600/400。
清单与 SHA256 保存于 `data/annotations/split.json`，重新运行不会静默改变划分。
随机划分不保证视频/场景级隔离，若后续发现连续帧或场景元数据，需另建分组验证实验。

## 启动与状态

```bash
python scripts/launch_baselines.py
```

启动器等待环境、预训练权重和完整上传的训练 ZIP，执行数据检查，再分别进行四卡
最大训练尺度 800 下的两个训练批次及验证冒烟测试，成功后并行启动两组正式训练。失败记录在
`logs/launcher_status.json`，不会静默切换到 CPU 或反复重启失败任务。
它会等待其他计算进程退出，不杀死他人任务。不要重复启动。

```bash
cat logs/launcher_status.json
tail -n 20 logs/rgb1600.log
tail -n 20 logs/rgb2000.log
nvidia-smi
```

## 权重与选优

结果分别保存在 `runs/rgb1600`、`runs/rgb2000`。

- `last.pth`：每轮原子替换保存，可恢复模型、EMA、优化器、AMP scaler、学习率进度和各 rank 随机状态。
- `weights_epoch_001.pth` … `weights_epoch_100.pth`：每轮保留轻量 EMA 推理权重，支持之后比较任意一轮；不含优化器，不用于精确续训。
- `epoch_005.pth` … `epoch_100.pth`：每 5 轮保留完整阶段权重。
- `best.pth`：仅验证版，按 400 张独立验证集 mAP@50–95 保存。
- `metrics.jsonl`、`status.json`：每轮损失、学习率、耗时，以及验证版 AP/各类 AP。
- `COMPLETE`：只有全部 100 轮成功结束才生成。

验证采用 COCO 101 点 AP，后处理全图最多 100 框；与比赛脚本是否完全一致仍待官方脚本核对。
全量版没有合法独立验证 AP，不能将训练过的 400 张当成验证集来挑“最佳模型”。
其保留的阶段权重可后续按比赛允许的评测流程比较。
续训保留随机状态，但高吞吐 cuDNN/多进程不承诺逐位确定性。

恢复单个任务示例（对应 GPU 空闲时）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --nproc_per_node=4 --master_port=29601 \
scripts/train_baseline.py --config configs/rgb1600.yml --resume runs/rgb1600/last.pth
```

代码、配置、划分清单和环境清单应另行备份到独立存储。本机阶段权重可用于续训，
但本机多个文件不能抵御整台服务器再次损坏。

## Git 管理

根目录仓库统一管理训练脚本、配置、依赖清单和 D-FINE 源码。上游来源和版本见
`D-FINE/UPSTREAM.md`；保留原许可证。`configs/ft_aug640_shared3.yml` 和
`configs/ft_aug800_shared3.yml` 保存六卡共享训练时的实际配置，每组 3 卡、
每卡 batch 3、PyTorch 显存上限 8.5 GiB。

数据、标注与划分文件、权重、训练输出、日志、图表、环境目录、备份和提交包由
`.gitignore` 排除。代码提交不能代替数据和权重的独立备份。远程仓库：<https://github.com/Acckion/aicomp>。

首次在新副本启用提交检查：

```bash
git config core.hooksPath .githooks
```

检查会拒绝数据目录、模型/媒体/压缩文件和超过 5 MiB 的暂存文件。
修改代码后可先查看 `git status` 与 `git diff`，仅提交代码和配置。
实验生成的配置与结果保存在被忽略的 `experiments/` 和 `runs/` 下；确认有效的
训练配置应另存到 `configs/` 后提交。环境通过已有 Miniconda AICOMP 环境运行。

自动后续流程使用 `scripts/next_baseline_stage.py`：当前验证微调完成后筛选收益，
再迁移到 2000 张全量训练、复测推理配置并生成 phase2 候选包。


### 夜间模型与上下文训练实验

DEIMv2-L/X使用官方COCO预训练检测权重，RGB输入、AICOMP12类、本地训练。首次准备运行 `python scripts/prepare_deimv2.py`；离线复核运行 `python scripts/prepare_deimv2.py --offline`。源代码revision及权重SHA256固定在该脚本，源代码/权重置于Git忽略目录。额外依赖见 requirements_deimv2.txt，继续使用Miniconda AICOMP环境。

启动 `python scripts/run_night_experiments.py --job l` 或 `--job x`；各自配置为configs/deimv2_l.yml、configs/deimv2_x.yml。先做峰值尺度训练和验证检查，成功才正式训练24轮。独立val400每轮评估mAP@50-95及类别AP；最后4轮关闭强增强。分类头适配12类，禁止套用COCO类别顺序。

启动 `python scripts/run_night_experiments.py --job context`，依次训练contextmix800与contextfull800，各12轮，同一权重和配置，区别为局部视图概率.35与0。约65%整图保留上下文；局部视图锚选择按小目标尺寸和GT空间密度加权。仅使用官方训练标注，不使用测试训练或模型集成。全部实验完成后保留配对报告，不按本地分数自动声称phase2收益。

`python scripts/plot_night_experiments.py`每分钟更新monitoring/night_experiments图表。三条队列各自记录后台进程、训练日志、失败原因和完成状态；已有未完成训练需显式处理，不能重复覆盖启动。

DEIMv2独立推理使用 `scripts/evaluate_variants.py --backend deimv2 --config configs/deimv2_l.yml --checkpoint <EMA推理权重> --size 640 --output <新输出目录> --single-method --method none`，X需改为对应配置。该backend自动使用官方ImageNet归一化、保持AICOMP类别0–11、重建尺度相关buffer。`scripts/after_night_models.py`可独立后台等待两候选完成并生成最佳/最终轮的整体与困难子集报告；不会自动提交。
