# Week7 Server Medium-Scale Run Plan

Week7 的服务器运行路线固定为 base-only first。当前主实验执行模型是 Qwen3-8B，这是执行 backbone 的配置，不改变 week6 已经收敛出的 structured risk current best family，也不重新打开 ranking family 搜索。官方 API 后端继续作为外部观察和小规模 cross-model 参考，不承担主实验吞吐。

这次路径修正属于服务器真实环境配置修正，不是方法变更。仓库默认的 Qwen3 本地配置现在指向当前服务器已通过 ModelScope 下载并验证可加载的目录：`/home/ajifang/models/Qwen/Qwen3-8B`。`configs/model/qwen3_8b_local.yaml` 中的 `model_name_or_path` 与 `tokenizer_name_or_path` 都应保持这个路径；如果后续换服务器，只调整这两个本地路径字段，不改方法配置、实验 family 或 summary 口径。

推荐的模型准备命令形态是：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /home/ajifang/models/Qwen/Qwen3-8B
```

这个命令只记录公开模型来源与服务器本地目录，不包含任何账号、token 或临时凭据。推理阶段继续使用 `local_files_only: true`，确保运行时只从服务器本地路径加载，不依赖在线 Hugging Face 访问。

Qwen3 默认可能输出 `<think>...</think>` 推理内容，这部分不进入方法叙事，而作为本地执行兼容问题处理。当前配置在 chat template 层设置 `enable_thinking: false`，三类 prompt 都要求只返回最终 JSON，同时 `src/llm/parser.py` 会在 pointwise、candidate ranking 和 pairwise 三条任务解析前统一去除残余 thinking block。因此服务器 smoke check 如果仍看到 raw response 中有 thinking 内容，parser 也应只把最终 JSON 交给后续评测。

当前服务器四域 processed 数据已经就位：`data/processed/amazon_beauty`、`data/processed/amazon_books_small`、`data/processed/amazon_electronics_small`、`data/processed/amazon_movies_small`。下一步仍然是继续服务器上的 backend check / smoke，而不是回头改数据或方法。

推荐执行顺序如下。第一，确认服务器 conda 环境、PyTorch、Transformers、Accelerate 与 CUDA 可用，并确认 `/home/ajifang/models/Qwen/Qwen3-8B` 存在。第二，执行 `python main_backend_check.py --model_config configs/model/qwen3_8b_local.yaml --status_path outputs/summary/week7_day1_backend_check.csv` 做最小加载与 schema 检查，确认 pointwise、candidate ranking、pairwise 三类 prompt 都能在同一 local-HF backend 下返回可解析结果。第三，执行 `RUN_SMOKE=1 bash scripts/week7_day1_server_backend_check.sh "$PWD"`，用三条 smoke config 验证完整入口。第四，smoke parse success 正常后，再使用 `python main_batch_run.py --batch_config configs/batch/week7_local_scale.yaml --run` 启动 Week7 local-scale batch。第五，等 local-scale 稳定后，再进入 `configs/batch/week7_medium_scale.yaml` 的 Beauty medium-scale handoff。

LoRA 不在 Week7 day5 强行启动。当前路线是先跑 Qwen3-8B base-only inference baseline；如果 Week8 或 reviewer-gap 结果显示任务确实需要适配，再用 PEFT 做 LoRA 或 QLoRA，并且只保存 adapter，不复制或重训整份 base model。推理阶段优先使用 Transformers 直接加载 base model 或 `base + adapter`；当吞吐成为主要瓶颈时，再考虑 vLLM 常驻服务。

当前 batch/registry 已经是 family-aware。`configs/batch/week7_medium_scale.yaml` 同时登记 pointwise、direct candidate ranking、structured risk current best rerank 和 pairwise，其中 structured risk 行显式记录 `method_family=structured_risk_family`、`method_variant=nonlinear_structured_risk_rerank`、`is_current_best_family=true`。这保证 week7 不再面对模糊的 uncertainty-aware rerank 集合，而是围绕 week6 已经收敛出的 current best family 做服务器端验证。
