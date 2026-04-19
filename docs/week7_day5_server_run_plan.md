# Week7 Server Medium-Scale Run Plan

Week7 的服务器运行路线固定为 base-only first。当前主实验执行模型已经从 Llama 3.1 8B Instruct 切换为 Qwen3-8B，这是执行 backbone 的切换，不改变 week6 已经收敛出的 structured risk current best family，也不重新打开 ranking family 搜索。官方 API 后端继续作为外部观察和小规模 cross-model 参考，不再承担主实验吞吐。

模型获取与加载路径统一走 ModelScope 到服务器本地目录。推荐把 Qwen3-8B 下载或同步到 `/home/ajifang/autodl-tmp/models/Qwen3-8B`，并让 `configs/model/qwen3_8b_local.yaml` 中的 `model_name_or_path` 与 `tokenizer_name_or_path` 指向该目录。`local_hf_backend.py` 继续使用 `local_files_only: true`，因此推理阶段只从服务器本地路径加载，不依赖运行时联网访问 Hugging Face。

推荐的模型准备命令形态是 `modelscope download --model Qwen/Qwen3-8B --local_dir /home/ajifang/autodl-tmp/models/Qwen3-8B`。如果服务器镜像里的 ModelScope CLI 版本不同，可以使用等价的 Python/CLI 下载方式，但最终约束不变：仓库配置只记录本地模型目录，不记录账号、token 或临时凭据。

Qwen3 默认可能输出 `<think>...</think>` 推理内容，这部分不进入方法叙事，而作为本地执行兼容问题处理。当前配置在 chat template 层设置 `enable_thinking: false`，三类 prompt 都要求只返回最终 JSON，同时 `src/llm/parser.py` 会在 pointwise、candidate ranking 和 pairwise 三条任务解析前统一去除残余 thinking block。因此服务器 smoke check 如果仍看到 raw response 中有 thinking 内容，parser 也应只把最终 JSON 交给后续评测。

推荐执行顺序如下。第一，确认服务器 conda 环境、PyTorch、Transformers、Accelerate 与 CUDA 可用，并确认 Qwen3-8B 的 ModelScope 本地目录存在。第二，执行 `main_backend_check.py --model_config configs/model/qwen3_8b_local.yaml` 做最小加载与 schema 检查，确认 pointwise、candidate ranking、pairwise 三类 prompt 都能在同一 local-HF backend 下返回可解析结果。第三，使用 `main_batch_run.py --batch_config configs/batch/week7_medium_scale.yaml --run` 启动 Beauty medium-scale batch，先跑 pointwise、direct candidate ranking 和 pairwise。第四，在 pointwise 结果完成校准并生成 calibrated uncertainty 后，再运行 structured risk current best rerank；该任务已经在 batch 清单中登记为 `is_current_best_family=true`，但依赖 direct rank prediction 与 calibrated uncertainty 先存在。第五，运行 `main_compare_baselines.py` 和 `main_compare_week7_medium_scale.py` 刷新 baseline matrix 与 medium-scale summary。

LoRA 不在 Week7 day5 强行启动。当前路线是先跑 Qwen3-8B base-only inference baseline；如果 Week8 或 reviewer-gap 结果显示任务确实需要适配，再用 PEFT 做 LoRA 或 QLoRA，并且只保存 adapter，不复制或重训整份 base model。推理阶段优先使用 Transformers 直接加载 base model 或 `base + adapter`；当吞吐成为主要瓶颈时，再考虑 vLLM 常驻服务。这个顺序可以避免过早引入训练复杂度，也能保证每一层新增能力都有明确实验理由。

当前 batch/registry 已经是 family-aware。`configs/batch/week7_medium_scale.yaml` 同时登记 pointwise、direct candidate ranking、structured risk current best rerank 和 pairwise，其中 structured risk 行显式记录 `method_family=structured_risk_family`、`method_variant=nonlinear_structured_risk_rerank`、`is_current_best_family=true`。这保证 week7 不再面对模糊的 uncertainty-aware rerank 集合，而是围绕 week6 已经收敛出的 current best family 做服务器端验证。

当前本地交付是 dry-run/status handoff，而不是伪造 GPU 运行结论。`outputs/summary/week7_day5_batch_status.csv` 用来检查输入、命令和依赖状态；`outputs/summary/week7_day5_medium_scale_summary.csv` 会在服务器真实运行和评测完成后吸收指标。若某一项失败，优先使用 `main_batch_run.py --batch_config configs/batch/week7_medium_scale.yaml --only_failed --run` 定点恢复，而不是重新手工串命令。
