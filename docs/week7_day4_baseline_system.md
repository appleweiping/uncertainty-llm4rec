# Week7 Baseline System Notes

Week7 的 baseline 层不再只把若干对照方法分散放在不同实验段落里，而是把三类证据放入同一张矩阵：uncertainty-source baseline、decision-formulation baseline、literature-aligned baseline。统一入口是 `main_compare_baselines.py`，核心聚合逻辑在 `src/analysis/aggregate_baseline_results.py`，默认输出为 `outputs/summary/week7_day4_baseline_matrix.csv`。

这张矩阵的作用不是制造一个更大的表，而是把不同 baseline 回答的问题区分清楚。uncertainty-source baseline 负责回答 uncertainty 如何定义以及 raw、calibrated、self-consistency、fused 等来源在不同任务粒度下是否稳定；decision-formulation baseline 负责回答在 pointwise、pairwise、candidate ranking 中，不使用 uncertainty 的同任务对照与当前 uncertainty-aware 主线之间的差异；literature-aligned baseline 负责回答当前方法是否已经可以放到接近文献范式的同候选集、同划分、同指标条件下比较。

模型身份也在这张矩阵中被显式记录。官方 API 组主要承担 cross-model 现象观察、外部黑盒参考和小规模 case study；本地 Hugging Face 小模型组承担后续主实验吞吐、批处理和系统比较；LoRA-adapted model 当前只保留接口身份，后续如果需要进入 base-only 与 base+adapter 对照，可以沿用 `model_family` 与 `adapter_path` 字段，而不需要重写 baseline schema。

当前 day4 的矩阵是结构化收口而不是服务器结论。它优先复用现有 compact evidence 产物：`week6_day3_estimator_compare.csv`、`week6_day4_decision_baseline_compare.csv` 和 `week7_day3_literature_baseline_summary.csv`。因此其中部分行来自 Beauty + Qwen 的 Part5 compact evidence，部分 literature-aligned baseline 行来自 Week7 的 local-HF identity schema。当前默认服务器主模型已经切换为 Qwen3-8B，这只是执行 backbone 变更，不影响 structured risk current best family 的方法定位。后续上服务器完成真实 local-HF 运行后，只需要重跑对应 summary 和 `main_compare_baselines.py`，同一张矩阵就可以吸收新的主模型结果。

这一天的边界同样需要保持清楚：不新增 ranking family，不把 local swap 或 fully fused 升级为默认主线，不把 pairwise 写成 candidate ranking 的替代主线。当前默认 ranking 主线仍然是 structured risk current best family；local swap 与 fully fused 继续作为 retained exploratory family 留在表和代码中；pairwise 继续作为机制层证据进入 baseline 体系。
