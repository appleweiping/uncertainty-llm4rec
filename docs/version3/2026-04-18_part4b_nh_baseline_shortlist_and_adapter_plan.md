# Part 4b / NH Baseline Shortlist 与接入方案

日期：2026-04-18

## 目标

这一轮不是做 baseline benchmark 全复现，而是补齐 Version 3 的 `baseline-aware` 防守层：

- 从 `baseline/NH` 中收敛出第一批真正值得接入的 baseline
- 明确 `主选 / 备选 / 暂缓`
- 把第一批 baseline 收成 minimal-change adapter 方案
- 保证它们能在同一 candidate setting 下输出 `performance + proxy confidence`

这里的“最小”指研究边界最小，而不是实现质量降低。我们仍然要求：

- 接口统一
- 输出稳定
- NH / NR 口径一致
- 结果文件可以直接进入后续总表

## 选型原则

本轮 shortlist 使用以下标准：

1. 优先 `candidate-based`，而不是开放式 free-generation
2. 优先能输出 `candidate scores` 或 `rank list`
3. 优先 NH 口径容易对齐的推荐论文
4. 优先 `minimal-change` 可以接入当前 grouped candidate setting 的工作
5. 不追求论文原始训练链路完整复现

## 文献依据

这轮 shortlist 结合了本地 `baseline/NH` 文件夹中的论文文件名，以及公开主来源检索结果。关键参考来源如下：

- SLMRec（ICLR 2025, OpenReview）
  - https://openreview.net/pdf/0744ad8ec274379fa6e5d41fc71d0f84dd405d71.pdf
- LLM-ESR（NeurIPS 2024 / OpenReview）
  - https://openreview.net/forum?id=xojbzSYIVS
  - https://github.com/liuqidong07/LLM-ESR
- CoVE（ACL 2025 Findings）
  - https://aclanthology.org/2025.findings-acl.651.pdf
- Aligning Large Language Models for Controllable Recommendations（ACL 2024）
  - https://aclanthology.org/2024.acl-long.443/

## 主选 baseline

### 1. CoVE

状态：主选

原因：

- 它天然是 candidate-aware / item-token aware 的 LLM recommender
- 论文核心思想是给 item 分配唯一 token，再在推荐空间内做分数比较
- 这和我们当前 `candidate_ranking` 主线非常容易对齐
- 最适合做 `score-based confidence proxy`

本轮接入方式：

- 不追求原论文完整训练或词表扩展工程
- 先实现一个 `CoVE-style candidate scorer`
- 在当前 candidate set 内，为每个 candidate 分配 synthetic item token
- 用同一组 candidate 上的 normalized scores 形成 `score_rows`

可直接构造的 proxy：

- `top1_score`
- `top2_score`
- `score_margin`
- `score_entropy`
- `proxy_confidence`

### 2. SLMRec

状态：主选

原因：

- 它是 sequential recommendation baseline，和当前项目的用户历史 -> 候选集预测设定很接近
- 它本质上是“把大模型知识蒸馏进小模型”的推荐框架，适合做 `small-model / score-based` baseline
- 即使不完整复现蒸馏训练，也可以稳定收成 embedding-based minimal adapter

本轮接入方式：

- 不追求完整 teacher-student 蒸馏
- 先实现一个 `SLMRec-style embedding scorer`
- 用用户历史构造 user representation
- 用 candidate title/meta 构造 item representation
- 用 cosine / dot product 输出 candidate scores

可直接构造的 proxy：

- `top1_score`
- `top2_score`
- `score_margin`
- `score_entropy`
- `proxy_confidence`

## 备选 baseline

### 3. LLM-ESR

状态：备选 / 第二批预留

原因：

- 它是对 sequential recommendation 的 enhancement framework，而不是单一、直接可抽出来的轻量 scorer
- 从论文定位看，更适合在 `adapter interface` 稳定之后作为第二批扩展
- 它的结构更适合“增强已有 SRS 模型”，不如 CoVE 和 SLMRec 适合先做最小 adapter

本轮处理方式：

- 不正式实现
- 在代码结构上预留第三个 adapter 的扩展位

## 暂缓 / 放弃

### Aligning Large Language Models for Controllable Recommendations

状态：暂缓

原因：

- 这篇更偏 `instruction alignment / controllability`
- 它不是最顺手的 minimal-change candidate scorer
- 第一批 baseline-aware 目标是快速建立 `performance + proxy confidence` 的同 setting 参照
- 对我们当前这一轮来说，它更像后续分析性补充，而不是最优先接入对象

## 本轮代码化方案

### 新增子系统

- `src/baseline/base.py`
- `src/baseline/adapters/cove_adapter.py`
- `src/baseline/adapters/slmrec_adapter.py`
- `src/baseline/io.py`
- `src/baseline/proxy.py`
- `src/baseline/eval.py`
- `main_baseline_eval.py`
- `main_baseline_confidence.py`

### 统一输入

优先输入：

- grouped candidate samples

兼容 fallback：

- 当前 pointwise rows

这样我们既能吃未来的 grouped candidate 文件，也能直接复用现有 `data/processed/.../test.jsonl`。

### 统一输出

固定输出目录：

- `outputs/baselines/{baseline_name}/{exp_name}/predictions/`
- `outputs/baselines/{baseline_name}/{exp_name}/metrics/`
- `outputs/baselines/{baseline_name}/{exp_name}/proxy/`
- `outputs/baselines/{baseline_name}/{exp_name}/logs/`

核心产物：

- `predictions/baseline_predictions.jsonl`
- `predictions/rank_rows.csv`
- `predictions/score_rows.csv`
- `metrics/ranking_metrics.csv`
- `proxy/proxy_results.csv`

### 统一代理信号

第一批统一支持：

- `top1_score`
- `top2_score`
- `score_margin`
- `score_entropy`
- `rank_gap`
- `proxy_confidence`

## 本轮边界

明确不做：

- calibration 全链路接入
- rerank 接入
- robustness 接入
- baseline 原论文训练链复现
- 第二批 baseline 正式实现

## 结论

当前最合理的 NH baseline 第一批接入是：

- 主选：`CoVE`
- 主选：`SLMRec`
- 备选：`LLM-ESR`
- 暂缓：`Aligning Large Language Models for Controllable Recommendations`

这个组合最符合 Version 3 当前目标：

- 研究边界收紧
- candidate setting 对齐
- 可稳定输出 score / rank
- 可直接构造 proxy confidence
- 不把 baseline 线扩成第二条主研究线
