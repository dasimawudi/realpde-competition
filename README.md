# RealPDE Competition 工作仓库

这是 RealPDE Competition Track 1 的实验整理仓库，包含我们用于复现实验、微调、打包和本地校验提交包的脚本与记录。

## 当前结论

截至 2026-08-30，Codabench 真实提交结果显示：

- `submission_cno_tke1200_bounds_rel00.zip`: `75.58455`
- 2026-08-29 的 UNet 后处理提交: `74.48384`

因此当前主线已经从 UNet 切回 CNO。UNet 在本地验证代理分数较高，但隐藏榜单上 Rel-L2、TKE、MVPE 都弱于 CNO，说明之前的 local final proxy 和验证集后处理有明显泛化风险。

## 仓库内容

- `tools/realpde_tke_finetune.py`：CNO 物理损失微调脚本。
- `tools/realpde_arch_finetune.py`：通用架构微调脚本，支持 UNet 等模型。
- `tools/realpde_calibrate_bounds.py`：本地评估与区间 bounds 扫描脚本。
- `tools/realpde_compare_architectures.py`：不同 baseline 架构对比脚本。
- `tools/realpde_ensemble_scan.py`：候选模型 ensemble 扫描脚本。
- `tools/realpde_feature_engineering.py`：从 `u/v/p` 构造速度、涡度、散度、时间差分、坐标等派生特征。
- `tools/realpde_feature_adapter_train.py`：在 CNO 前训练一个 residual feature adapter，默认冻结 CNO 主体。
- `tools/realpde_h5_feature_adapter_train.py`：直接读取官方 HDF5 `u/v` 轨迹的 feature-adapter 训练脚本，适合 `/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830/` 这种数据布局。
- `tools/realpde_h5_cno_weight_scan.py`：在 HDF5 验证 split 上扫描两个 CNO checkpoint 的权重插值/外推。
- `tools/realpde_pack_cno_template.py`：复用已验证的 CNO 提交模板，替换 `model.pth` 并调整 bounds。
- `tools/realpde_pack_feature_adapter.py`：把 feature-adapter checkpoint 打成 Codabench 可上传 zip。
- `docs/submission_log.md`：提交与候选包记录。
- `docs/feature_engineering_plan.md`：下一轮特征工程实验方案与远端命令。

## 不进 Git 的内容

数据集、checkpoint、提交 zip、官方 `RealPDEBench/` checkout 都被 `.gitignore` 排除。原因是这些文件体积较大，且有些是比赛发布/远程训练产物，不适合直接推到 GitHub。

本地提交包仍保存在工作目录中；README 只记录文件名和用途。

## 环境

远程训练机当前使用官方兼容镜像：

```text
pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

官方比赛页声明该镜像为评测 Docker image。CNO 包只依赖 `torch`、`numpy` 和随包 vendored 的 `rpde_baselines/cno.py`。

## 推荐下一次提交

如果下一次只能提交一个包，当前建议优先试：

```text
submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip
```

它是 CNO `tke4100` → `cont600` 权重外推 `lambda=2.15` 后，再混入 20% Rel-L2/MVPE 微调方向的单模型包。相比纯外推版，它在本地验证上更均衡：Rel-L2、MVPE、SPS 改善，TKE 只小幅回退。`nobench` 版本刻意不启用 `cudnn.benchmark`，避免 Codabench 冷启动计时被卷积算法搜索拉高。

用户已在 2026-08-30 手动提交该包，并反馈真实榜分为 `75.94193`。下一轮冲分建议以它对应的 checkpoint 为起点，先试 `docs/feature_engineering_plan.md` 中的 frozen feature adapter。

2026-08-31 在 HDF5 直读 split 上发现 frozen feature adapter、低学习率续训、向 P0 baseline 权重插值都不稳；当前只产生一个低风险 bounds-only 候选：

```text
submission_cno_tke4100_lam215_microa020_abs0075_rel015_nobench_20260831.zip
```

它与 `75.94193` 包使用同一个预测模型，只把不确定性区间改成 `abs=0.0075, rel=0.015`。

保守备选：

```text
submission_cno_tke4100_bounds_abs0075_rel000_flat_20260829.zip
```

它是 CNO `tke4100` checkpoint 的 flat-clean 单模型版本，最接近已在榜上表现最好的 CNO 简洁路线。

## 注意

Codabench 页面说明当前阶段使用 Starting Kit v9，`sps_score` 已改为线性映射；`final_score` 的组合方式不公开，starting kit 只保证五个子分的计算一致。因此本仓库中的本地 proxy 只能作为调参参考，不能作为真实 leaderboard 分数承诺。
