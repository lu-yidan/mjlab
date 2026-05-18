# G1 Recovery 快捷脚本

在仓库根目录执行（或先 `cd` 到 `mjlab`）：

```bash
chmod +x scripts/recovery/*.sh
```

## 常用命令

| 脚本 | 作用 |
|------|------|
| `list_runs.sh` | 列出所有 recovery 训练 run 与最新 checkpoint |
| `train_smoke.sh` | 128 env × 20 iter 冒烟 |
| `train_polish_full.sh` | 4096 env × 6000 iter 姿态打磨（从零） |
| `train_robust_resume.sh` | 从 polish checkpoint 续训抗扰动 |
| `play_gui.sh` | 本地交互播放（有显示器用 native，否则 viser） |
| `play_video.sh` | 无头录视频到 `logs/.../videos/play/` |
| `extract_frames.sh` | 从 mp4 抽关键帧 png |
| `kill_gpu_jobs.sh` | 清理占用 GPU 的旧 train/play 进程 |

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MJLAB_RUN_DIR` | 最新 robust run | run 目录名或绝对路径 |
| `MJLAB_CHECKPOINT` | `model_5999.pt` | checkpoint 文件名 |
| `MJLAB_NUM_ENVS` | `4096`（训练）/ `1`（play） | 并行环境数 |
| `MJLAB_VIEWER` | `auto` / `viser` | play 查看器 |
| `MJLAB_VIDEO_LENGTH` | `240` | 录视频帧数 |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU 编号 |

## 示例

```bash
# 列出训练记录
./scripts/recovery/list_runs.sh

# 播放当前最佳 robust checkpoint（默认 model_5999）
./scripts/recovery/play_gui.sh

# 指定 checkpoint 并录视频
MJLAB_CHECKPOINT=model_5000.pt ./scripts/recovery/play_video.sh
./scripts/recovery/extract_frames.sh

# 从零开始姿态打磨训练
./scripts/recovery/train_polish_full.sh

# 从 polish@4400 续训抗扰动
MJLAB_LOAD_RUN=2026-04-15_11-07-35_late_phase_polish_full_v1 \
  MJLAB_RUN_NAME=my_robust_v3 \
  ./scripts/recovery/train_robust_resume.sh
```

完整流程、奖励说明与下一步优化见：[G1 Recovery 工作流文档](../../docs/source/g1_recovery_workflow.md)。
