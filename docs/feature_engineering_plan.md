# Track 1 特征工程计划

当前保底路线是 CNO 单模型；2026-08-30 用户手动提交的 `submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip` 已拿到 `75.94193`。下一步不应直接大幅改 backbone，而是先做可回退的输入侧特征工程。

## 核心想法

比赛输入仍是前 20 帧的 `u, v, p`，输出仍是后 20 帧的 `u, v, p`。特征工程不改变提交接口，只在模型内部从输入帧派生更多信息：

- `speed = sqrt(u^2 + v^2)`：速度大小。
- `kinetic_energy = 0.5 * (u^2 + v^2)`：局部动能量级。
- `du_dt, dv_dt`：相邻输入帧速度变化，提示短期趋势。
- `vorticity ≈ dv/dx - du/dy`：旋涡/剪切结构。
- `divergence ≈ du/dx + dv/dy`：局部压缩/扩张迹象。
- `strain_magnitude`：速度梯度形变强度。
- `x_coord, y_coord, t_coord`：显式坐标和时间位置。

## 为什么用 adapter 而不是直接改 CNO 输入层

CNO 官方 checkpoint 的第一层只接受 3 通道。如果把输入直接改成 13 通道，就很难完整继承当前已经拿分的 CNO 权重。新脚本采用 residual adapter：

```text
原始 u/v/p -> 派生 13 个特征 -> 小 MLP 投影回 3 通道 -> 原 CNO -> 后 20 帧预测
```

adapter 最后一层初始化为 0，所以训练第 0 步等价于原模型；如果 adapter 没学到东西，验证分不会天然崩掉。这一点对当前每天提交次数有限的局面很重要。

## 推荐第一轮远端实验

如果数据目录是直接 HDF5 轨迹，例如 `/home/chyfuture/RealPDE_data/p0ab_real_h5_20260830/`，优先使用 `realpde_h5_feature_adapter_train.py`，它不依赖 RealPDEBench 的 `Foil` loader。

先同步本仓库的特征工程分支到训练机：

```bash
source /etc/network_turbo
if [ ! -d /root/autodl-fs/realpde_repo/.git ]; then
  git clone https://github.com/dasimawudi/realpde-competition.git /root/autodl-fs/realpde_repo
fi
cd /root/autodl-fs/realpde_repo
git fetch origin
git checkout codex/feature-engineering
git pull
cp tools/realpde_feature_engineering.py /root/autodl-fs/realpde_runs/
cp tools/realpde_feature_adapter_train.py /root/autodl-fs/realpde_runs/
cp tools/realpde_h5_feature_adapter_train.py /root/autodl-fs/realpde_runs/
cp tools/realpde_pack_feature_adapter.py /root/autodl-fs/realpde_runs/
```

在 AutoDL/GPU 服务器上，从当前最强 CNO checkpoint 开始：

```bash
cd /root/autodl-tmp/realpde/RealPDEBench
python /root/autodl-fs/realpde_runs/realpde_feature_adapter_train.py \
  --checkpoint <path-to-current-best-cno-model.pth> \
  --run-name cno_feature_adapter_frozen_$(date +%Y%m%d_%H%M) \
  --num-update 1200 \
  --eval-interval 100 \
  --batch-size 12 \
  --test-batch-size 64 \
  --adapter-lr 3e-4 \
  --point 1.0 \
  --mse 0.05 \
  --tke 0.08 \
  --temporal 0.04 \
  --grad 0.02 \
  --p-zero 0.01
```

HDF5 直读版示例：

```bash
python /repo/tools/realpde_h5_feature_adapter_train.py \
  --real-root /data/p0ab_real_h5_20260830 \
  --checkpoint /runs/current_best/model.pth \
  --realpdebench-root /third_party \
  --out-dir /runs/cno_h5_feature_adapter_frozen_YYYYMMDD_HHMM \
  --updates 1200 \
  --eval-interval 100 \
  --batch-size 8 \
  --test-batch-size 32 \
  --adapter-lr 3e-5 \
  --adapter-delta 0.02
```

如果 adapter-only 比 iteration 0 有提升，再试更小学习率联训 backbone：

```bash
python /root/autodl-fs/realpde_runs/realpde_feature_adapter_train.py \
  --checkpoint <当前最强CNO checkpoint> \
  --run-name cno_feature_adapter_unfreeze_$(date +%Y%m%d_%H%M) \
  --num-update 600 \
  --eval-interval 100 \
  --batch-size 8 \
  --test-batch-size 64 \
  --adapter-lr 1e-4 \
  --base-lr 5e-8 \
  --train-base
```

## 打包

训练结束后，用保底 CNO zip 作为模板复制 `rpde_baselines/cno.py`：

```bash
python /root/autodl-fs/realpde_runs/realpde_pack_feature_adapter.py \
  --checkpoint /root/autodl-fs/realpde_runs/<run_name>/model_best.pth \
  --template-zip /root/autodl-fs/realpde_runs/submission_cno_tke4100_lam215_microa020_abs0075_rel0075_nobench_20260830.zip \
  --out /root/autodl-fs/realpde_runs/submission_cno_feature_adapter_YYYYMMDD.zip \
  --bound-abs 0.0075 \
  --bound-rel 0.0075
```

提交前必须先在远端做一次随机输入 smoke test 和 validation proxy 检查。只有本地验证至少不低于 iteration 0，才值得消耗 Codabench 当日提交名额。
