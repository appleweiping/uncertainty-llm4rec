# Week7.5 Compare Bridge Notes

week7.5 第四天的重点不是提前产出 LoRA framework 的真实训练结论，而是先把它在 compare 层的身份、字段、角色和后续接入方式固定下来。这样做的原因很直接：week5、week6、week7 已经形成了三层 baseline 坐标系，如果 week7.5 的 trainable framework 不能自然进入这套坐标系，那么后续服务器真实结果回来之后，项目就会再次陷入“结果有了，但 compare 语义还得临时补”的状态。day4 的 compare bridge 正是为了解决这个问题。

当前 compare bridge 的核心原则有三条。第一，`structured risk current best family` 必须被正式固定为 `strongest_handcrafted_baseline`，而不是模糊地写成“以前的方法”或“当前 best”。因为 week7.5 的 trainable framework 只有在正面对上这个 strongest hand-crafted baseline 时，后续训练复杂度才有可辩护的意义。第二，`direct ranking` 必须继续保留为 `same_task_reference`，用来回答 trainable framework 相比 non-uncertainty 同任务基线是否真的提升。第三，`literature-aligned ranking baselines` 保持 `literature_or_task_aligned_reference` 的位置，它们回答的不是 uncertainty 是否有意义，而是这个 trainable framework 是否能在相同 schema 下站到更接近外部文献范式的对照面前。

在实现上，day4 新增了两层 compare 产物。第一层是 `week7_5_framework_compare.csv` 与 `week7_5_framework_compare.md`，它们直接围绕 week7.5 的 framework compare 语义展开，显式记录 `baseline_layer`、`current_role`、`compare_status`、`training_stage_role` 等字段。第二层是 `week7_5_baseline_matrix.csv` 与 `week7_5_baseline_matrix.md`，它们负责把 framework compare 重新桥接回 `week7_day4_baseline_matrix.csv` 那套更大的 baseline matrix 语义中。前者更接近 framework 自己的闭环视角，后者更接近整个项目 baseline 体系的统一视角。

对后续服务器真实结果来说，这个桥接层的意义很明确。一旦 `main_lora_train_rank.py -> main_eval_lora_rank.py -> main_compare_framework.py` 真实跑通，新的 adapter 路径、framework ranking metrics 和 compare 状态不会停留在局部 csv，而会顺着 bridge 直接进入现有 baseline matrix。也就是说，服务器真实闭环回来的不是一张孤立的 framework compare 表，而是一套可以直接挂到既有三层 baseline 坐标系上的新证据层。这样进入 week8 时，项目面对的就不是“一个尚未接线的 trainable 分支”，而是一个已经有清晰 compare 身份的中心方法对象。

day5 的 speed upgrade 则是在这个 compare bridge 之上补上执行层约束。它回答的不是“要不要继续保留 framework compare”，而是“在不删 pointwise / pairwise / calibration / baseline 的前提下，framework compare 和 baseline bridge 后续如何承受更大的 ranking 负载”。因此 compare bridge 与 speed upgrade 不是两件独立的事情：compare bridge 固定的是方法身份和 baseline 身份，speed upgrade 固定的是这些身份进入 week8 之后依然可执行的吞吐路径。只有这两层同时成立，week8 才不至于把一个方法上已接线、但执行上仍然会塌掉的 framework 扔进更大矩阵里。
