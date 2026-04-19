# week5-week6 缺漏审计与 week7 承接建议

目前我已经把 version pony 的 week5 和 week6 工作计划、对应日报以及 summary 产物系统过了一遍。整体判断是，Part5 这条线已经完成了计划内的方法闭环，现在不再是 toy，也不是空壳，已经足够支撑“方法部分成立”和“论文初稿实验可写”这一级别的推进。week5 这边，多任务数据层、prompt/parser、三任务 inference、三任务 eval、multitask summary 都已经有对应实现和产物；week6 这边，ranking decision family、pairwise-to-rank、estimator compare、same-task baseline、Part5 final results 和 final summary 也都已经落地。所以现在的问题已经不是 week5/week6 做没做完，而是如何把当前 Beauty + Qwen 的小规模闭环，继续升级成更强、更完整、能直接应对 reviewer 追问的论文级证据系统。

我对当前缺口的判断是，计划内交付基本齐了，但顶会级证据还需要继续压实。第一，pairwise coverage 是现在最硬的短板。现有结果已经说明 pairwise-to-rank 这条机制线有积极信号，尤其在 overlap subset 上，uncertainty-aware weighted aggregation 相比 direct reference 和 plain aggregation 都有更好的排序表现；但它当前只覆盖了 6/30 个 ranking events，supported_event_fraction 只有 0.2，所以现阶段它更适合写成“机制层证据”而不是“主决策替代线”。后续更值得投入的是 coverage 扩展，以及 overlap 和 expanded 两种评测范围下的诚实对照，而不是继续发明更复杂的 pairwise aggregation 公式。换句话说，pairwise 当前的价值不是它已经可以取代 candidate ranking，而是它证明局部偏好机制能够为主任务提供额外信号，只是这个信号还需要更充分的覆盖支撑。

第二，论文级图表还没有真正系统化沉淀。现在结果表、日报和 summary 都基本齐了，但 figure pack 还没有正式收出来。下一步至少应该补三类图：pointwise 诊断图，用来支持 uncertainty elicitation 与 calibration 的基础可信度；多任务 family compare 图，用来说明 structured risk、local swap、fully fused、pairwise-to-rank 这些 family 在统一口径下各自的位置；以及 pairwise coverage 图，把当前支持率、overlap 表现和 expanded fallback 表现明确画出来。这里要注意，图表不是为了把日报可视化，而是为了服务论文叙事。pointwise 图回答“uncertainty 是否可信”，family compare 图回答“为什么 current best family 是现在这条线”，coverage 图回答“pairwise 机制线的边界在哪里”。跨域、跨模型和 noisy robustness 的图可以放到后面更大规模阶段再统一做，不需要现在把所有图一次性摊开。

第三，当前证据规模还主要停留在 Beauty + Qwen 的小闭环。这个阶段已经足够说明结构成立，但还不够说明结论稳。后续优先级应该先放在扩数据域，而不是先把五个官方 API 全铺开。更合理的下一步，是把当前 Part5 的最小完整链路复制到四个正式数据域：Movies、Beauty、Books 和 Electronics，每个域先控制在 100 个 sample 的可管理规模内，并统一使用 DeepSeek 做主实验通道。这样做的好处是，实验规模不会一下子失控，但能直接回答“这是不是 Beauty + Qwen 的局部现象”。多模型可以轻量补，但官方 API 更适合作为小规模黑盒观察和 cross-model 现象验证，不应该继续承担主实验吞吐。主证据应该从可复现、可批处理、可扩展的统一后端里来，官方 API 更像外部先验和 case-level 观察窗口。

第四，baseline 体系还差 literature-aligned baseline。week6 day4 补齐的是 Part5 内部非常重要的 same-task baseline，这一步已经很关键，因为它让 uncertainty-aware 方法有了同任务、同输入、同指标的直接对照。但它还不是最终论文防守意义上的强 baseline。后面还需要把更接近推荐、uncertainty-aware reranking、LLM4Rec candidate ranking 相关文献里的可比 baseline 接进来，并放到统一候选集、统一 split、统一指标下做公平对照。这个部分不应该在 week6 里抢跑，因为 week6 的职责是先把多任务方法闭环做实；但从 week7 往后，literature-aligned baseline 必须逐步进入实验体系，否则论文在 reviewer 面前会容易被质疑“只和自己设计的内部 baseline 比”。

第五，当前方法主线应该先冻结，不适合继续无限开 family。structured risk 现在已经足够作为 current best ranking family 往后推进；local swap 和 fully fused 应该继续保留在表里和代码里，作为 retained exploratory family，不删、不抹掉，但也不再抢后面 week7/week8 的默认主实验资源。fully fused 只有在更大样本、更高 coverage、更复杂候选集，或者跨域 robustness 场景里真正稳定显出优势时，再考虑升级优先级。这个策略的意义是避免过早放弃复杂 family 的潜在价值，同时也避免把主线写散。对论文来说，清楚地说明“当前主线是什么、保留线是什么、触发升级条件是什么”，比不断试新公式更有说服力。

因此，我现在对 week7 的承接理解是：不需要再怀疑 Part5 有没有完成，它已经完成了；真正要做的是把 Part5 从 Beauty + Qwen 小规模方法闭环，推进成更稳的证据系统。最合理的顺序是，先冻结 week6 的方法主线，再补论文级图表；然后做 pairwise coverage 扩展和本地 batch 后端；再把 Part5 的最小完整链路复制到 Movies、Beauty、Books、Electronics 四个数据域，每域先跑 100 个 sample，统一用 DeepSeek 作为可控主通道；之后再用少量官方 API 做 cross-model 观察；最后补 literature-aligned baseline 和 robustness 矩阵。这样推进会更像顶会导向的节奏，不是继续堆公式，而是把已经完成的方法闭环压实成真正可答 reviewer 的证据系统。

需要额外提醒的是，当前代码和文档状态也要继续保持同步。README、Part5 finalize 脚本和最终收口日报需要一起进入版本管理，避免出现“论文总结写了，但入口脚本和项目说明没跟上”的脱节。后续每完成一个阶段性研究动作，都应该继续保留两层记录：README 只写项目级方向和可复现入口，不写得过细；Paper/pony2 下的日报或阶段报告则用中文密集段落记录真实判断、边界、贡献和下一步。这种记录方式既能保护具体细节不被过度暴露，也能让整个 version pony 的研究推进保持可追溯、可复盘、可继续扩展。
