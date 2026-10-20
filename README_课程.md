# Inspection Lab：基于 anomalib 的轻量工业缺陷检测实验室

课程项目：深度学习与图像处理。最终采用的二次开发目标是 **保持原始四项检测指标不变，减少模型部署包体积**，同时提供本地中文可视化界面。PCA 和显存优化实验保留作为有完整结果的探索对照。

**已验证结果：[无损存储优化报告](course/无损存储优化报告.md)。** 9 模型包从 119.03 MiB 降到 29.00 MiB（减少 75.63%）；单模型包减少 8.18%–9.60%。恢复文件逐字节一致，9 组测试的图像/像素 AUROC 与 F1 全部不变。收益属于部署包存储，不是 GPU 显存或推理加速。

上游：[open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)，固定版本 `lib/v2.5.1`，提交 `fcf9ffa7382c1bfdd85e65bf7b2ecae0f6a9c1af`。上游源码和 Apache-2.0 许可证保留，课程扩展位于 `course/`。本项目不是 anomalib 官方产品，也不宣称提出了新的 PCA 算法。

## 本地启动

当前电脑的项目位于 `E:\GPT\anomalib-course`。双击 **启动实验室.cmd**，浏览器打开 `http://127.0.0.1:8765`。也可在项目目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-course.ps1 gui
```

启动窗口需要保持运行。服务只监听本机，不需要把图片上传到云端。若服务已运行，启动入口会直接打开现有界面。

- **实验总览**：历史指标、样本完整性、缺陷热力图、CSV 导出。
- **图像检测**：选择已训练模型，上传 PNG/JPEG，查看异常得分、预测区域。得分不是概率；图片应与模型的训练类别一致。
- **无损部署**：组合包与单模型体积对比、一致性验证、下载和恢复命令。
- **对比实验**：选择数据集、baseline/PCA 96/PCA 192、随机种子、CUDA/CPU；运行任务并查看日志。
- 同条件对比只接受新版实验：相同配置、相同数据划分、相同压缩前特征库哈希。旧实验保留，但不会与新版混算。

## 数据规模与实验划分

| 数据 | 正常训练图 | 原始测试图 | 缺陷标注 | 用途 |
|---|---:|---:|---:|---|
| hazelnut_toy | — | — | 22 | 56 张演示图；部署测试 |
| bottle | 209 | 83 | 63 | 正式类别 |
| cable | 224 | 150 | 92 | 正式类别 |
| capsule | 219 | 132 | 109 | 正式类别 |

三个完整类别共 **1,017 张图像 + 264 张标注**，加演示集共 1,073 张图像。不是完整 15 类 MVTec AD。

实验将每个类别的原始测试部分按固定种子划成互不重叠的验证集和最终测试集（比例 0.5，具体数量见每次报告）。验证集用于上游后处理和阈值校准，测试集仅用于评估；PCA 只拟合正常训练样本的 coreset 特征。每次保存 `split_manifest.json`，记录图像、标签和 SHA256。这个协议不是论文的完整测试集协议，不能直接与论文排行榜比较。

下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\run-course.ps1 download-full
```

来源是 [Kaggle 的 MVTec AD 公开镜像](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)。官方链接失效时采用这个重新打包的镜像；不能称其与官方压缩包字节一致。下载器校验 ZIP CRC、PNG 解码和预期数量，并记录每个文件 SHA256。网络中断后可重跑，已校验的文件跳过。数据遵循 MVTec AD 的 CC BY-NC-SA 4.0，限学术、非商业使用；代码仓库不包含数据或模型权重。

## 二次开发内容

1. `compact_patchcore.py`：保留上游 coreset 选点，在选出的正常特征库上拟合 PCA，将 384 维压缩到 96 或 192 维；查询特征使用同一均值和投影矩阵。
2. `experiment.py`：统一配置、独立划分、样本清单、压缩前特征库哈希、训练耗时、推理计时和存储统计。
3. `gui.py` 与 `web/`：本地中文 GUI，后台单任务执行，模型选择、上传检测、可视化与导出。
4. `download_mvtec.py`：按类别下载完整数据、校验和续传。
5. `lossless_package.py`：原始 checkpoint ZIP 数据区去重、无损压缩与逐字节恢复。
6. `audit_lossless.py` / `report_lossless.py`：模型包体积、恢复完整性、9 组原始检测指标复验及报告。
7. `exact_search.py` / `optimize_exact.py`：未采用的显存探索；完整测试出现耗时增加，未接入默认检测。

所有方法固定 ResNet18 ImageNet 权重、layer2+layer3、224×224 输入、batch=1、coreset=1%、9 个邻居。这里的 baseline 是**相同轻量配置下未修改的上游 PatchCore**，不是上游 WideResNet 默认配置或论文最优成绩。

压缩模型的额外构造通过隔离随机状态、复制原始骨干权重，确保不会改变后续 coreset 随机选点。`protocol_version=paired-coreset-v2` 标识此修正；旧版记录没有这个字段，属于早期探索数据。

## 最终采用方案的衡量标准

- **质量约束**：恢复文件 SHA-256 与原文件相同，所有 state_dict 张量相同，9 组原始检测的四项 AUROC/F1 全部相同。
- **改进指标**：包含恢复信息的模型部署包总字节数；分别报告单模型和多模型包。
- **运行方式**：部署时先无损恢复，再走原始推理路径，不改变训练和检测。打包约 2.06 秒、恢复单模型约 0.104–0.134 秒，是新增的一次性部署成本。

## 早期 PCA 探索的衡量标准

- **主要目标**：`search_state_bytes = memory_bank_bytes + projection_bytes` 下降。不能只报特征库缩小而遗漏投影开销，也不能把这个指标称为总显存。
- **质量约束**：同时报告 image/pixel AUROC 和 F1。压缩可能损失精度，需要公开所有预先选定类别的结果。
- **速度观察**：固定第一张测试图，预热 10 次、同步计时 30 次，记录模型前向中位耗时。包含骨干、投影、近邻检索和原始异常图；不包含读图、预处理、阈值后处理或界面显示。单次笔记本计时受温度与其他程序影响，不据此宣称稳定加速。
- **完整模型**：另报 checkpoint 大小。骨干占据大部分模型空间，特征库减少 75% 不意味着整个模型减少 75%。

单种子结果保留在 [初步实验报告](course/实验报告.md)。新的 [多种子实验报告](course/多种子实验报告.md) 比较 baseline / PCA96 / PCA192，在三个类别、种子 42/43/44 下共 27 组，报告均值、样本标准差和配对变化。多个划分共享数据，不构成独立外部验证。

## 复现实验

```powershell
.\.venv\Scripts\python.exe course\experiment.py --dataset bottle --method baseline --seed 42
.\.venv\Scripts\python.exe course\experiment.py --dataset bottle --method pca96 --seed 42
.\.venv\Scripts\python.exe course\experiment.py --dataset capsule --method pca192 --seed 42
.\.venv\Scripts\python.exe course\test_lab.py
```

三个类别 × 三种方法 × 三个种子的 27 组 PCA 对照已经完成。PCA 对部分类别有质量损失，不作为“指标不退化”要求下的最终方案。多模型无损存储不调整模型参数或阈值，恢复后仍是原始模型。

不应为了“精度提升 1%”反复用测试标签调参。对于本课程，明确的存储节省、可解释的精度权衡、可复现的流程和可用界面已经是可衡量的开发内容。

## 换电脑安装

需要 Windows、Git 和 uv；NVIDIA 驱动应支持锁定的 CUDA 运行时。克隆后运行：

```powershell
git clone https://github.com/DragonLi1004/anomalib-inspection-lab.git
cd anomalib-inspection-lab
powershell -ExecutionPolicy Bypass -File .\setup-course.ps1
```

私有仓库需使用获得授权的 GitHub 账号。安装器用 uv 管理 Python 3.12.15 与上游 `uv.lock`，所有运行时、缓存、数据和输出放在项目目录。初次依赖下载较大，尤其 PyTorch。安装后可使用已下载的模型与数据离线训练和预测。官方资源或镜像不可用时脚本会明确报错。

`course/outputs/`、`course/data/`、`course/models/`、`.runtime/`、`.cache/` 和 `.venv/` 不上传。不要加载不可信来源的 `.ckpt`，界面仅选择本项目生成的模型。

## 仓库工作流

上游 CI 工作流作为参考保存在 `docs/upstream-workflows/`，不在本课程仓库自动运行；GitHub Actions 已关闭。这样不会误启动上游的发布或大规模测试任务。

## 当前验证范围

20 项检查通过（核心功能 7 项、统计汇总 4 项、显存探索回归 3 项、无损包与下载接口 6 项）：投影距离/正交性质、训练均值、模型初始化与随机状态一致性、checkpoint 重载后的查询投影、本地接口和文件访问范围，以及样本标准差、配对变化、缺失组与不匹配特征库的处理。三个完整类别的训练、评估与 GUI 展示已验证。浏览器文件上传授权未通过，因此文件选择到上传的交互链路尚未完成端到端验证；上传功能已实现，不能称这一步已验证通过。

## 批量实验与多种子汇总

```powershell
# 查看固定矩阵中哪些实验还需运行，不启动训练
.\.venv\Scripts\python.exe course\experiment_suite.py --dry-run
# 顺序执行；相同源码版本已有完整模型和指标的实验自动复用
.\.venv\Scripts\python.exe course\experiment_suite.py
# 所有 27 组完成且配对校验通过后，导出表格、报告与科学图
.\.venv\Scripts\python.exe course\summarize_experiments.py
.\.venv\Scripts\python.exe course\test_summary.py
```

批量脚本逐次保存状态和独立日志到 `course/outputs/suites/`，遇到失败会停止；重新执行可续跑。批量脚本与 GUI 的单任务执行分别管理进程，使用批量脚本时不要同时通过 GUI 启动另一个训练，以免争用显卡。

GUI「对比实验」页新增 CUDA 多种子汇总：只有相同源代码、数据划分、配置与 coreset 的 baseline/PCA96/PCA192 三组齐全，才计入该类别的统计；每类种子数单独显示。只有一个种子时标准差显示为空，不以零冒充稳定性。

复用 seed=42 的早先运行意味着计时日期、温度和负载并不受控。因此耗时表只作描述，不证明稳定加速。重点看含投影的存储变化与配对精度变化。

## 无损模型包复现

保持项目环境由 uv 管理，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe course\audit_lossless.py
.\.venv\Scripts\python.exe course\report_lossless.py
.\.venv\Scripts\python.exe course\test_package.py
```

审计脚本需要已经完成的 9 个原始模型（3 类别 × 3 种子）和对应数据。每次创建新输出目录，组合包、单模型包、恢复模型及日志都保存在 `course/outputs/lossless-*`。模型包不上传 GitHub，代码与匿名汇总报告上传；可在 GUI「无损部署」页复制具体恢复命令。恢复完成后直接使用原始 PatchCore 加载，推理无需安装额外压缩依赖。
