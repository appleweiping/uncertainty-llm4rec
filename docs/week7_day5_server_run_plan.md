# Week7 Server Medium-Scale Run Plan

Week7 的服务器运行路线固定为 base-only first。当前主实验模型是 Llama 3.1 8B Instruct，模型保存在服务器本地路径或 HF cache 中，本地电脑只维护代码和配置，服务器通过 pull 最新代码后执行 batch。官方 API 后端保留为外部观察和小规模 cross-model 参考，不再承担主实验吞吐。

推荐执行顺序如下。第一，确认服务器 conda 环境、PyTorch、Transformers、Accelerate 与 CUDA 可用，并确认 `configs/model/llama31_8b_instruct_local.yaml` 中的模型路径命中服务器本地缓存。第二，执行 `main_backend_check.py` 做最小加载与 schema 检查，确认 pointwise、candidate ranking、pairwise 三类 prompt 都能在同一 local-HF backend 下返回可解析结果。第三，使用 `main_batch_run.py --batch_config configs/batch/week7_medium_scale.yaml --run` 启动 Beauty medium-scale batch，先跑 pointwise、direct candidate ranking 和 pairwise。第四，在 pointwise 结果完成校准并生成 calibrated uncertainty 后，再运行 structured risk current best rerank；该任务已经在 batch 清单中登记为 `is_current_best_family=true`，但依赖 direct rank prediction 与 calibrated uncertainty 先存在。第五，运行 `main_compare_baselines.py` 和 `main_compare_week7_medium_scale.py` 刷新 baseline matrix 与 medium-scale summary。

LoRA 不在 Week7 day5 强行启动。当前路线是先跑 base-only inference baseline；如果 Week8 或 reviewer-gap 结果显示任务确实需要适配，再用 PEFT 做 LoRA 或 QLoRA，并且只保存 adapter，不复制或重训整份 base model。推理阶段优先使用 Transformers 直接加载 base model 或 `base + adapter`；当吞吐成为主要瓶颈时，再考虑 vLLM 常驻服务。这个顺序可以避免过早引入训练复杂度，也能保证每一层新增能力都有明确实验理由。

当前 batch/registry 已经是 family-aware。`configs/batch/week7_medium_scale.yaml` 同时登记 pointwise、direct candidate ranking、structured risk current best rerank 和 pairwise，其中 structured risk 行显式记录 `method_family=structured_risk_family`、`method_variant=nonlinear_structured_risk_rerank`、`is_current_best_family=true`。这保证 week7 不再面对模糊的 uncertainty-aware rerank 集合，而是围绕 week6 已经收敛出的 current best family 做服务器端验证。

当前本地交付是 dry-run/status handoff，而不是伪造 GPU 运行结论。`outputs/summary/week7_day5_batch_status.csv` 用来检查输入、命令和依赖状态；`outputs/summary/week7_day5_medium_scale_summary.csv` 会在服务器真实运行和评测完成后吸收指标。若某一项失败，优先使用 `main_batch_run.py --batch_config configs/batch/week7_medium_scale.yaml --only_failed --run` 定点恢复，而不是重新手工串命令。
