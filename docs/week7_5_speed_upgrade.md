# Week7.5 Speed Upgrade

week7.5 第五天的目标不是重新设计方法，也不是为了追求速度去删掉研究内容，而是把“提速但不减内容”正式写进执行层。当前项目的主瓶颈被明确定位在 `candidate ranking` 路径上，而不是 pointwise、pairwise、calibration 或 baseline 体系本身。这个判断并不意味着其它层不重要，恰恰相反，它意味着这些层都必须被保留：pointwise 继续提供 diagnosis 与 calibrated uncertainty，pairwise 继续提供 mechanism evidence，baseline 继续提供三层 compare 约束，而真正需要优化的是 ranking-heavy 的执行方式。

因此，week7.5 的提速策略必须先写清楚边界。第一，不能通过删 pointwise、删 pairwise、删 calibration、删 baseline 来换速度。第二，不能把提速写成“以后再说”的口头承诺，而必须固定成可追踪、可回写、可进入 week8 入场条件的工程产物。第三，提速的优先级不是所有地方一起动，而是先围绕 ranking 路径的四个问题组织：输出契约是否过长、batch_generate 是否真正生效、重复模型加载是否过多、以及后续更大规模时是否已经具备 shard/resume/serving 的迁移路径。

当前第一条执行升级是收紧 ranking 输出契约。对于 week7.5 之后的主实验，默认 ranking 输出应当向最小必要 JSON 靠拢，把长 reasoning 从默认执行路径剥离出来，只在 case study 或 debug setting 下再打开。这样做不是减少实验内容，而是把“内容”与“输出冗余”区分开。研究内容仍然完整保留：candidate ranking 主任务照跑，pointwise / pairwise / calibration / baseline 继续存在，但 ranking 生成不再默认负担大段解释文本。

第二条执行升级是把 `batch_generate` 从“配置里写了 batch size”推进成“执行层已经确认 batch 真正吃到了”。week7.5 的速度路径明确要求检查 `main_rank.py` 和 `src/llm/local_hf_backend.py` 之间是否确实走到了 `batch_generate()`，而不是逻辑上仍在逐条 `generate()`。这件事必须在 week8 前写成显式检查项，因为如果不先确认 local-HF ranking 真正具备 batching 行为，那么后面扩大样本和扩大域时，速度问题只会被成倍放大。

第三条执行升级是减少重复模型加载，并把 `shard / resume / registry` 明确接入 ranking-heavy 路径。当前 week7.5 已经要求 LoRA framework 走 `startup_check -> train -> eval -> compare` 的 closure path，day5 则进一步要求这个 closure path 在执行层具备 shardable 与 resumable 的准备状态。也就是说，后续不论是 framework ranking eval、structured risk compare 还是 week8 的更大矩阵，都应尽量避免整轮重跑，而是优先通过 shard 和 registry 恢复局部失败的 ranking 任务。

第四条执行升级是为更大规模阶段预留常驻服务化接口，但不在 week7.5 提前切换整套后端。当前执行主线仍然是 `Transformers + local HF backend`，这是与已有 Qwen3-8B 服务器路径兼容的最稳选择。但从 week7.5 起，项目必须明确写下：一旦 ranking volume 超出 Beauty 单域 medium 阶段，后续就要评估是否切到更常驻的 serving 模式，例如 vLLM。这里的重点是“预留切换条件”，而不是在 week7.5 把服务化本身强行做完。

这条 speed upgrade 路径与 framework compare 和 baseline bridge 的关系也必须写清楚。它不是与 compare 并行存在的附属说明，而是 compare 能否进入 week8 的前提之一。换句话说，week8 不只是要求 trainable framework 已经有 Beauty 单域最小结果，也要求项目已经知道如何在不删研究内容的前提下继续扩张 ranking 路径。因此 day5 之后，speed upgrade 必须与 `week7_5_framework_compare.csv / .md`、`week7_5_baseline_matrix.csv / .md` 和 framework manifest 一起存在。只有这样，week8 才不是把一个尚未处理吞吐瓶颈的方法直接扔进更大的验证矩阵，而是把一个已经明确了方法角色与执行边界的中心对象正式扩展出去。

因此，week7.5 结束时服务器上的第一次真实闭环顺序也要固定下来：先做 `startup_check`，确认训练输入、评估输入、adapter 输出目录与 compare 路径无误；再跑 Beauty 单域 LoRA train；训练后立即做 framework eval，生成标准 ranking metrics；再刷新 framework compare 与 baseline bridge；最后刷新 speed upgrade plan，把当前的 ranking 提速路径、week8 入场条件和 shard/resume/serving 准备情况一起回写。这条顺序的意义不在于多跑一步命令，而在于让 week7.5 在结束时同时拥有方法闭环、compare 闭环和执行闭环。
