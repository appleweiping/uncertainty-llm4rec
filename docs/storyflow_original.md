下面我给你设计一个可以往 **ICLR/NeurIPS/KDD/WWW/RecSys 顶会主会**冲的完整论文项目。核心不是“把 uncertainty、LLM4Rec、popularity bias、data pruning 缝在一起”，而是把它们统一成一个新的问题：**在生成式推荐里，confidence 不是一个事后解释分数，而是会改变曝光、反馈和未来训练分布的行动变量。** 换句话说，LLM 说“我很确定推荐这个 title”，这句话本身会让系统更愿意曝光它；一旦曝光后被点击，未来训练数据又进一步强化这种确定性。因此推荐场景里的 confidence calibration 不能只问“confidence 与当前 ground truth 是否相关”，而要问：**这个 confidence 到底是在表达真实偏好，还是在表达 popularity prior、语言模型熟悉度、历史曝光偏差、catalog grounding 确定性，甚至训练噪声？**

---

## 0. 项目总题目与一句话主张

我建议题目定成：

**Do LLM Recommenders Know When They Recommend? Causal Confidence Calibration for Generative Recommendation**

方法名可以叫：

**CURE-Rec: Causal Uncertainty-Regularized Generative Recommendation**

或者更有记忆点一点：

**TRUCE-Rec: Triangulated and Exposure-Calibrated Uncertainty for Generative Recommendation**

论文的一句话 thesis：

> In generative LLM-based recommendation, confidence should be calibrated not only to offline correctness, but to counterfactual user utility under exposure; otherwise high-confidence recommendations become a popularity-driven feedback amplifier, causing overconfident hallucinations, long-tail underconfidence, and echo-chamber reinforcement.

这条线的好处是它能把你提到的所有问题自然串起来：LLM 生成 title 是否正确、它是否知道自己正确、错误是不是因为低 confidence、popular item 是否天然高 confidence、high confidence 是否导致 echo chamber、uncertainty 能否用于 pruning noisy training data。它们不是六个松散问题，而是一个闭环：**generation → confidence → exposure → feedback → training → future confidence**。

这个切入与 Bryan Hooi 组的方向非常贴：他们组主页强调 trustworthiness、factuality、awareness of what models do not know、bias、distribution shift，也关注 foundation models 与 graphs / e-commerce / recommendation 这类 structured data 的结合；ConfTuner 用 proper scoring rule 训练 LLM 口头表达 confidence，ConfMAD 进一步说明不合适的 confidence 会让系统固执坚持错误或过早收敛，TrustGen 则强调动态、可扩展的 trustworthiness benchmark。你的项目可以把这些思想迁移到推荐，但不是简单迁移：推荐的 confidence 会改变曝光分布，这是 QA 或 reasoning calibration 里没有的核心差异。([bhooi.github.io](https://bhooi.github.io/))

---

## 1. 为什么这个问题现在值得做：已有工作空缺在哪里

生成式推荐已经从传统“打分—排序—重排”的 pipeline，转向让 LLM 或 generative retriever 直接从 item pool 生成推荐结果；LREC-COLING 2024 的 survey 明确把 LLM-based generative recommendation 描述为用 LLM 直接生成推荐，而不是只把 LLM 当 feature extractor。TIGER 代表了 Semantic ID 路线：把 item 编成语义 token tuple，然后自回归生成目标 item ID；GenRec / instruction-following recommendation / BIGRec 则更接近自然语言或 grounded item generation。BIGRec 特别指出，很多 LLM4Rec 只在 restricted candidate set 上评估，不能真实反映 all-rank / actual item grounding 能力，并提出“language space → recommendation space → actual item space”的 grounding。([aclanthology.org](https://aclanthology.org/2024.lrec-main.886/))

但是这些工作大多默认：只要生成结果更准，系统就更好。问题是，LLM 生成推荐 title 时，它还会隐含或显式给出一种“确定性”：log probability 高、beam margin 大、自我回答“yes, I am confident”、多次 sampling 一致、或者 verbal confidence 很高。这个 confidence 可能和 ground truth 正相关，也可能只是 item 流行度、title 常见度、LLM 预训练熟悉度、catalog grounding 容易度的代理变量。

近两年已经有一些 uncertainty-aware LLM4Rec 相关工作。WWW 2025 的 “Uncertainty Quantification and Decomposition for LLM-based Recommendation” 提出估计 LLM 推荐的 predictive uncertainty，并分解成 recommendation uncertainty 与 prompt uncertainty；AAAI 2026 的 GUIDER 把 hallucination 归因到 data uncertainty 与 model uncertainty，并用单次 inference 的 logit evidence 做 uncertainty-guided reranking；2026 的 “Uncertainty-aware Generative Recommendation” 已经明确提出 generative recommendation 存在 “uncertainty blindness”，用 uncertainty-weighted reward、difficulty-aware optimization、explicit confidence alignment 改进 Semantic-ID 式 generative recommendation。([openreview.net](https://openreview.net/forum?id=lhFcbb5q48))

所以如果只是做“给 LLM4Rec 加 uncertainty score”，很容易被认为已经有人做了。你要超过它们，需要把问题提升一层：**已有工作主要把 uncertainty 当作模型内部 reliability 或 optimization signal；你的项目把 confidence 视为推荐系统闭环里的 causal exposure variable。** 你的关键区别是：

第一，你做的是 **free-form title generation + catalog grounding + confidence**，而不只是假设 item 已经是固定 Semantic ID。因为真实 LLM4Rec 很可能直接生成 “The Legend of Zelda: Breath of the Wild” 这种 title，输出可能正确、近似、歧义、重复、out-of-catalog、popular-but-wrong、semantically plausible but unobserved。这个 evaluation 比 ID generation 更贴近 LLM generative recommendation 的本质。

第二，你不只问 calibration 到 offline next-click，而问 **counterfactual calibration under exposure**：如果系统真的把这个 item 曝光给用户，confidence 是否对应用户会接受它的概率，而不是对应“这个 item 在历史日志中更常见”。这自然引出 popularity bias、echo chamber、long-tail underconfidence。

第三，你把 uncertainty 用于 training data pruning，但不是粗暴“高 uncertainty 就删”。在推荐中，高 uncertainty 可能是噪声，也可能是长尾真实兴趣、冷启动用户、偏好多样性、未来兴趣转移。你的创新点应是 **uncertainty decomposition for training triage**：区分 label-noise uncertainty、epistemic uncertainty、aleatoric preference uncertainty、catalog grounding uncertainty、popularity-induced confidence。

---

## 2. 核心 observation：从 generative recommendation 的输出开始做

你最终 task 是 generative：输入用户历史 item title，输出一个推荐 title。因此 observation 不能只做传统 top-k ranking score，而要围绕“生成结果本身”设计。

设用户历史为
\[
H_u = [i_1, i_2, ..., i_t],
\]
每个 item 有 title、category、brand、description、popularity。LLM 接收自然语言 prompt：

> User has interacted with: [title1], [title2], ..., [titlet].
> Generate the next item title the user is most likely to interact with.
> Then answer: Is your recommendation likely to be correct? Yes or No.
> Confidence: an integer from 0 to 100.

模型输出：

\[
\hat y_u = \text{generated title}, \quad a_u \in \{\text{Yes}, \text{No}\}, \quad c_u \in [0,1].
\]

然后你把 generated title grounding 到 catalog：

\[
\hat i_u = g(\hat y_u),
\]

其中 \(g\) 可以是 exact match + normalized fuzzy match + embedding nearest neighbor + LLM judge + metadata disambiguation。ground truth 可以是 held-out next item \(i^+_{u,t+1}\)，也可以扩展成未来 \(m\) 个 interactions \(Y_u^+ = \{i^+_{t+1},...,i^+_{t+m}\}\)，避免“用户没点不代表不喜欢”的 implicit feedback 问题。

最基础的 correctness：

\[
z_u = \mathbf{1}[g(\hat y_u) \in Y_u^+].
\]

如果模型回答 Yes 且 confidence 是 \(c\)，则预测正确概率 \(p_u=c\)；如果它回答 No 且 confidence 是 \(c\)，则预测正确概率可以设成 \(p_u=1-c\)，或者更稳妥地直接让模型输出 “Probability this recommendation is correct: p%”。这样你就能把 generative recommendation 变成一个二分类 calibration probe：**模型生成了一个 title，然后判断这个 title 是否正确，并给 confidence；看这个 confidence 与 ground truth 的相关性。**

这一步非常关键，因为它直接回答你的问题：“LLM may generate correct output, but is it sure with the result?” 你会得到四个象限：

1. **Correct & Confident**：理想推荐。
2. **Correct & Uncertain**：模型知道得不够，可能是 long-tail underconfidence。
3. **Wrong & Uncertain**：正常失败，uncertainty 可用于 abstention 或 exploration。
4. **Wrong & Confident**：最危险，可能是 hallucination、popularity prior、echo-chamber amplifier。

这四象限会成为整篇论文最有解释力的 observation figure。

---

## 3. Observation 指标：不要只画 ECE，要画“confidence 的来源”

传统 calibration 只看 Expected Calibration Error：

\[
\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{n} \left|\text{acc}(S_b)-\text{conf}(S_b)\right|.
\]

但推荐场景更复杂。你需要设计一组新指标，最好能成为论文贡献之一。

### 3.1 Title Grounding Calibration

LLM 输出 title，不一定能对应真实 catalog item。你需要先衡量：

\[
\text{GroundHit} = \mathbf{1}[g(\hat y) \neq \varnothing],
\]

\[
\text{GroundAmbiguity} = 1 - \left(s_1 - s_2\right),
\]

其中 \(s_1, s_2\) 是 generated title 到最近和第二近 catalog title 的相似度。很多 LLM 可能高 confidence 生成一个“听起来合理但不存在”的 item，尤其在电影、游戏、书籍这种 title 空间里非常常见。这个叫 **grounding overconfidence**。

### 3.2 Verbal-Outcome Calibration

把 verbal confidence 和 correctness 对齐：

\[
\text{VO-ECE} = \sum_b \frac{|S_b|}{n}|\mathbb{E}[z|S_b]-\mathbb{E}[c|S_b]|.
\]

同时报告 Brier score：

\[
\text{Brier}=\frac{1}{n}\sum_u(c_u-z_u)^2.
\]

这与 ConfTuner 的思想呼应：ConfTuner 用 tokenized Brier score 训练 LLM 口头表达 confidence，并证明其作为 proper scoring rule 能激励模型报告真实正确概率；但你的 target 不是通用 QA correctness，而是 grounded generated recommendation correctness。([arxiv.org](https://arxiv.org/abs/2508.18847))

### 3.3 Correct-but-Uncertain / Wrong-but-Confident

定义两个更直观的 risk 指标：

\[
\text{CBU}_\tau = P(c < \tau \mid z=1),
\]

\[
\text{WBC}_\tau = P(c > \tau \mid z=0).
\]

CBU 高说明模型对正确答案不敢确定，常见于长尾 item、冷启动用户、偏好突然转移；WBC 高说明模型在错误答案上过度自信，常见于 popular item、语义 generic item、title hallucination。

### 3.4 Popularity-Confidence Coupling

你最想看的 observation 是：popular item 是否天然高 confidence，non-popular 是否天然低 confidence。不能只画 \(c\) 和 popularity 的 correlation，因为 popular item 本来可能更容易成为 ground truth。要做控制变量回归：

\[
c_u = \beta_0 + \beta_1 z_u + \beta_2 \log \text{Pop}(\hat i_u) + \beta_3 \text{Mainstream}(u) + \beta_4 \text{HistEntropy}(u) + \beta_5 \text{GroundConf}(\hat y_u) + \epsilon.
\]

这里 \(\beta_2\) 是关键。如果控制 correctness、用户 mainstreamness、history entropy、grounding confidence 后，\(\beta_2\) 仍显著为正，就说明 confidence 里有 popularity prior。

你还可以定义：

\[
\text{Tail Underconfidence Gap} = \mathbb{E}[c | z=1, \hat i \in \text{Head}] - \mathbb{E}[c | z=1, \hat i \in \text{Tail}].
\]

如果 correct head item 的 confidence 明显高于 correct tail item，说明模型不是“不知道正确不正确”，而是“不敢相信长尾正确”。

这个问题与推荐系统已有的 popularity bias / calibration literature 可以自然连接。Steck 的 calibrated recommendation 已指出 accuracy-oriented recommender 容易过度聚焦用户主兴趣，忽略小众兴趣；后续 popularity-bias 工作也讨论了 popular items 被过度推荐、long-tail visibility 被压制，以及 popularity bias 与 miscalibration / fairness 的联系。([dl.acm.org](https://dl.acm.org/doi/epdf/10.1145/3240323.3240372?utm_source=chatgpt.com))

### 3.5 Confidence-Induced Echo Risk

你的项目最创新的 observation 是：**confidence 不只是预测变量，它会进入 exposure policy。**

定义一个简单部署策略：

\[
\pi_{\text{conf}}(i|u) \propto \exp(s(u,i)+\lambda c(u,i)),
\]

其中 \(s(u,i)\) 是推荐 utility score，\(c(u,i)\) 是 confidence。随着 \(\lambda\) 增大，高 confidence item 获得更多曝光。如果 confidence 与 popularity 耦合，那么 exposure distribution 会向 head item 收缩。你可以用 offline simulation 测：

\[
\text{ExposureGini}, \quad \text{TailCoverage}, \quad \text{CategoryEntropy}, \quad \text{KL}(P_{\text{rec category}}||P_{\text{user history category}}).
\]

更进一步，做多轮 feedback loop：

\[
D^{(t+1)} = D^{(t)} \cup \{(u,i): i \sim \pi^{(t)}, \text{ click} \sim P_{\text{user}}(click|u,i)\},
\]

每轮重新微调或更新 lightweight adapter / reranker。观察：confidence-only policy 是否让 mean confidence 上升，但 true utility、diversity、tail coverage 下降。这就是你论文里的 **Confidence Amplification Loop**。

ConfMAD 的洞察可以被漂亮地类比过来：在 multi-agent debate 里，不恰当的 confidence 会让 agent 固执坚持错误或过早收敛；在推荐系统里，不恰当的 confidence 会让系统固执曝光 head/popular item，并通过用户反馈让错误确定性变成训练事实。([arxiv.org](https://arxiv.org/abs/2509.14034))

---

## 4. 论文的核心发现应该长什么样

你最终希望 observation 部分得出以下几类发现。注意这些是要通过实验验证的 hypotheses，不要在论文里先当成结论写死。

**Finding 1: LLM recommenders are not uniformly overconfident; they are selectively overconfident on familiar/popular titles and underconfident on long-tail correct titles.**
这比“LLM overconfidence”更推荐系统化。LLM 在 popular title 上可能 confidence 高，不管对不对；在 tail title 上即使生成正确，也可能低 confidence。

**Finding 2: Wrong high-confidence recommendations are not random errors; they concentrate on semantically generic and high-popularity items.**
比如用户历史有若干小众 RPG 游戏，模型高 confidence 推荐一个 mainstream blockbuster 游戏；不是完全无关，但不是真实 next item。这类错误最容易形成 echo chamber，因为它们“看起来合理”。

**Finding 3: Post-hoc self-verification is more popularity-biased than generation probability.**
模型生成后再问 “Is this correct? confidence?”，它可能根据 title 熟悉度和语义流畅性给高 confidence，而不是根据用户偏好证据。这个 finding 很有意思，因为它直接挑战“让 LLM 自我评估即可”的方法。

**Finding 4: Naively pruning high-uncertainty training samples hurts long-tail recommendation.**
因为 high uncertainty 混合了 noisy label 和 rare-but-valuable preference。真正有效的是 uncertainty decomposition：只 prune likely-noise，不 prune epistemic hard positives 或 tail positives。

**Finding 5: Confidence-guided exposure without causal calibration increases apparent certainty while reducing diversity and long-tail coverage.**
这就是 echo chamber 的证据：系统越来越 confident，但不是越来越懂用户，而是越来越陷入 head-item attractor。

这些 finding 如果做实，会比单纯提出一个 loss 更顶会，因为它开辟了一个问题空间。

---

## 5. 方法框架：CURE-Rec / TRUCE-Rec

### 5.1 总体框架

输入用户历史 \(H_u\)，模型生成 \(K\) 个 candidate titles：

\[
\{\hat y_{u,k}\}_{k=1}^K \sim \pi_\theta(y|H_u).
\]

每个 title 被 grounding 到 catalog item：

\[
\hat i_{u,k}=g(\hat y_{u,k}).
\]

对每个 candidate，系统估计一个 uncertainty vector，而不是单一 confidence：

\[
\mathbf{U}(u,\hat i)=
[
U_{\text{gen}},
U_{\text{ground}},
U_{\text{pref}},
U_{\text{epi}},
U_{\text{pop}},
U_{\text{noise}}
].
\]

含义分别是：

\(U_{\text{gen}}\)：生成分布不确定性，来自 length-normalized log probability、beam margin、semantic entropy、sample disagreement。
\(U_{\text{ground}}\)：title 到 catalog item 的 grounding uncertainty。
\(U_{\text{pref}}\)：用户偏好本身的 aleatoric uncertainty，比如历史兴趣分散、短期兴趣转移、多类别混合。
\(U_{\text{epi}}\)：模型 epistemic uncertainty，比如 prompt perturbation、adapter ensemble、MC dropout、不同 LLM disagreement。
\(U_{\text{pop}}\)：confidence 中由 popularity prior 解释的部分。
\(U_{\text{noise}}\)：训练样本可能是 noisy interaction / accidental click / exposure artifact 的概率。

最后输出：

\[
\hat y, \quad \hat i, \quad c_{\text{cal}}, \quad \mathbf{U}, \quad \text{decision}.
\]

decision 可以是 recommend、diversify、ask clarification、abstain、或者 route to retrieval model。

---

## 6. 模块一：Triangulated Confidence Elicitation

不要只相信 verbal confidence。你要把 confidence 分成五类来源：

**Self-verbalized confidence**：模型直接输出 “confidence: 80%”。
**Token likelihood confidence**：生成 title token 的平均 log probability。
**Margin confidence**：top-1 title 与 top-2 title 的 log-prob margin。
**Semantic consensus confidence**：多次 sampling 后，grounded items 是否聚成同一个 catalog item / same category。
**Counterfactual stability confidence**：轻微扰动用户历史、打乱非最近 item、mask 一个 popular history item 后，推荐是否稳定。

然后训练一个 calibrator：

\[
c_{\phi}(u,\hat i)=h_{\phi}(c_{\text{verb}}, c_{\text{ll}}, c_{\text{margin}}, c_{\text{sem}}, c_{\text{stab}}, \text{GroundConf}, \text{Pop}, \text{Mainstream}(u)).
\]

关键是 \(h_\phi\) 不只输出 calibrated confidence，还要输出 decomposition：

\[
c_{\phi}=c_{\text{pref}} + c_{\text{pop}} + c_{\text{ground}},
\]

其中 \(c_{\text{pref}}\) 才是你真正想用于推荐决策的“偏好置信度”，\(c_{\text{pop}}\) 是 popularity-driven confidence，\(c_{\text{ground}}\) 是因为 title 容易匹配 catalog 而带来的确定性。

这里可以借鉴 ConfTuner，但要做推荐版改造。ConfTuner 的 tokenized Brier score 是为 verbal confidence 设计的 proper scoring rule，不需要 ground-truth confidence 或 proxy confidence；你的 Rec-ConfTuner 可以让 LLM 输出 confidence bin token，例如 `<conf_0.7>`，然后用 grounded correctness 或 graded relevance 作为 target。([arxiv.org](https://arxiv.org/abs/2508.18847))

推荐版 loss：

\[
\mathcal{L}_{\text{RecBrier}}
=
\sum_{m=1}^{M}
\left(
p_{\theta}(\text{conf}=b_m|H_u,\hat y)
-
\mathbf{1}[r_u \in b_m]
\right)^2,
\]

其中 \(r_u\) 可以是 binary correctness，也可以是 graded target：

\[
r_u =
1 \cdot \mathbf{1}[\hat i=i_{t+1}]
+
\alpha \cdot \mathbf{1}[\hat i \in Y_{t+2:t+m}]
+
\beta \cdot \text{SemanticRel}(\hat i,Y_u^+).
\]

这样你避免了 implicit feedback 里“没点不等于不喜欢”的问题。

---

## 7. 模块二：Causal Popularity Disentanglement

这是项目的灵魂。

推荐日志不是随机样本。popular item 更常被曝光、更常被点击、更常出现在训练数据中，也更常出现在 LLM 预训练语料里。于是 LLM 的 confidence 可能满足：

\[
C \leftarrow \text{Popularity} \rightarrow \text{Training Frequency} \rightarrow \text{Model Familiarity}.
\]

同时：

\[
C \rightarrow \text{Exposure} \rightarrow \text{Click} \rightarrow \text{Future Training Data}.
\]

所以 calibration target 不能只是：

\[
C \approx P(Z=1|H,\hat i),
\]

而应该是：

\[
C \approx P(R=1 | H, \hat i, do(E=1)),
\]

即：如果我们真的把 item 曝光给用户，用户会接受它的 counterfactual probability。

你可以提出 **Counterfactual Confidence Calibration**：

\[
\forall b,g:\quad
\mathbb{E}[C|C \in B_b, G=g]
\approx
\mathbb{E}[R^{do(E=1)}|C \in B_b, G=g],
\]

其中 \(G\) 是 popularity group、user mainstreamness group、category group。实际估计可用三种近似：

第一，使用 chronological heldout future interactions，扩大正样本窗口。
第二，使用 inverse propensity / doubly robust correction，如果数据中有曝光或 position 信息。
第三，构造 matched counterfactual pairs：同类别、同语义相似度、不同 popularity 的 item pair，测试 confidence 是否因 popularity 改变。

训练上加一个 popularity residual loss：

\[
\mathcal{L}_{\text{pop-res}}
=
\left\|
\text{Corr}
\left(
C - \hat R,
\log \text{Pop}(\hat i)
\mid z,\text{Mainstream}(u),\text{Category}
\right)
\right\|^2.
\]

或者用 adversarial disentanglement：从 confidence residual representation 里预测不出 popularity bucket。

\[
\min_{\theta,\phi}\max_{\psi}
\mathcal{L}_{\text{rec}}
+
\lambda \mathcal{L}_{\text{cal}}
-
\gamma \mathcal{L}_{\text{adv-pop}},
\]

其中 adversary \(D_\psi\) 试图从 residual confidence 表征预测 head/mid/tail，主模型通过 gradient reversal 去掉 popularity-only signal。

注意不要把 popularity 完全去掉。popular item 对 mainstream 用户可能确实 relevant。因此约束目标不是 \(C \perp Pop\)，而是：

\[
C \perp Pop \mid R, \text{Mainstream}(u), \text{Category}, \text{GroundConf}.
\]

这点非常重要，否则 reviewer 会说你牺牲准确率做反 popularity。

---

## 8. 模块三：Risk-Aware Preference Optimization

训练 generative recommender 时，不应该把所有 wrong outputs 都同等惩罚。一个低 confidence 的 plausible wrong answer，和一个高 confidence 的 popular hallucination，风险不同。你可以设计 uncertainty-aware preference objective：

\[
\mathcal{L}
=
\mathcal{L}_{\text{SFT/DPO}}
+
\lambda_1 \mathcal{L}_{\text{RecBrier}}
+
\lambda_2 \mathcal{L}_{\text{pop-res}}
+
\lambda_3 \mathcal{L}_{\text{overconf}}
+
\lambda_4 \mathcal{L}_{\text{tail-underconf}}.
\]

其中：

\[
\mathcal{L}_{\text{overconf}}
=
\mathbb{E}
[
\mathbf{1}[z=0]\cdot c^2 \cdot h_{\text{exposure}}(\hat i)
],
\]

高 confidence 错误越危险，尤其如果 item 是 head item、曝光会进一步强化它，就惩罚更大。

\[
\mathcal{L}_{\text{tail-underconf}}
=
\mathbb{E}
[
\mathbf{1}[z=1, \hat i \in \text{Tail}]
\cdot
\max(0, \tau-c)^2
].
\]

这个 loss 专门防止“正确长尾推荐低 confidence”。这比普通 calibration 更贴近推荐系统公平性和探索价值。

Preference optimization 也可以改成 pairwise：对于同一用户，正样本 \(i^+\)、hard negative \(i^-\)。如果 \(i^-\) 是 popular 且模型高 confidence，就增加 margin：

\[
\mathcal{L}_{\text{conf-DPO}}
=
-\log \sigma
\left(
\beta[
\log \pi_\theta(i^+|H)
-
\log \pi_\theta(i^-|H)
]
-
\eta c(i^-)
+
\rho c(i^+)
\right).
\]

直觉是：**模型不仅要把正样本排在负样本前，还要降低对高风险负样本的确定性，并提高对真实长尾正样本的确定性。**

---

## 9. 模块四：Uncertainty-Guided Training Data Triage，不是 naive pruning

你提到 “prune training based on uncertainty due to noise?” 这是很有潜力的应用，但必须做得细。推荐数据里的高 uncertainty 至少有四种含义：

1. **Noise**：误点、偶然点击、爬虫、曝光诱导、重复购买异常、title mapping 错。
2. **Hard but valid**：长尾真实兴趣、兴趣转移、跨域偏好。
3. **Aleatoric ambiguity**：用户本身兴趣很散，多个 item 都合理。
4. **Epistemic insufficiency**：模型没见过类似用户或 item，但更多数据可改善。

所以不能按 loss 或 entropy 直接删。你要估计 noise posterior：

\[
N(u,i)
=
P(\text{label noisy}|H_u,i,\mathbf{U}).
\]

特征包括：

\[
N = f(
\text{high NLL},
\text{ensemble disagreement},
\text{low semantic support},
\text{low neighbor-user support},
\text{abnormal timestamp},
\text{catalog ambiguity},
\text{future contradiction},
\text{popularity-exposure anomaly}
).
\]

训练权重：

\[
w(u,i)
=
(1-N(u,i))
\cdot
(1+\alpha U_{\text{epi}}(u,i)\cdot S_{\text{support}}(u,i))
\cdot
(1-\beta U_{\text{pop-noise}}(u,i)).
\]

含义是：

高 noise posterior：降权或 prune。
高 epistemic uncertainty 但有 semantic / neighbor support：保留甚至加权，因为这是 hard positive。
高 popularity-induced noise：尤其谨慎，因为它可能是被曝光推出来的点击，而不是真偏好。
高 aleatoric uncertainty：不要删，改成 soft label / listwise target。

实验上你可以做两种：

第一，synthetic noise：随机替换一定比例 next item，分别替换为 head item、tail item、same-category item，测试 pruning 是否恢复性能。
第二，real noise：用 timestamp anomaly、低停留时间、低 rating、一次性点击、重复 title ambiguity 做弱噪声标签。

对比方法：

naive high-loss pruning、entropy pruning、small-loss curriculum、no pruning、你的 decomposed triage。

关键结果应该是：naive uncertainty pruning 会提高 head accuracy 但损害 tail coverage；你的 triage 同时提升 NDCG/ECE/noise robustness/tail coverage。

---

## 10. 推理时策略：confidence 不是越高越该推

传统做法可能是：

\[
\text{score}(u,i)=\hat r(u,i).
\]

confidence-aware naive 做法是：

\[
\text{score}(u,i)=\hat r(u,i)+\lambda c(u,i).
\]

你的方法应该是：

\[
\text{score}(u,i)
=
\underbrace{\mathbb{E}[R|u,i,do(E=1)]}_{\text{counterfactual utility}}
-
\lambda
\underbrace{\text{Risk}(u,i)}_{\text{overconf / grounding / hallucination}}
-
\mu
\underbrace{\text{EchoRisk}(u,i)}_{\text{pop-confidence exposure}}
+
\nu
\underbrace{\text{InfoGain}(u,i)}_{\text{safe exploration}}.
\]

其中：

\[
\text{EchoRisk}(u,i)
=
c_{\text{pop}}(u,i)
\cdot
\text{Pop}(i)
\cdot
(1-\text{DiversityGain}(u,i)).
\]

这表达一个核心思想：如果 confidence 主要来自 preference evidence，那可以用；如果 confidence 主要来自 popularity prior，就不能让它无限增加曝光。

部署策略可以有三种：

**Recommend**：\(c_{\text{pref}}\) 高，grounding 确定，echo risk 低。
**Diversify**：多个 candidate 都 plausible，aleatoric uncertainty 高。
**Ask / abstain**：grounding uncertainty 高或 overconf risk 高。
**Explore**：epistemic uncertainty 高、tail item plausible、echo risk 低。

这样论文会从 offline calibration 延伸到 decision-level trustworthiness，层次更高。

---

## 11. 实验设计：从 observation 到 framework

### 11.1 数据集

建议至少三类：

**Movie / Book / Game**：title 语义强，LLM 熟悉度差异大，容易观察 popular title high confidence。MovieLens、Amazon Books、Steam、Goodreads 都适合。
**E-commerce**：Amazon Beauty、Sports、Toys、Electronics，适合 LLM4Rec sequential setting。
**Business / POI / Yelp**：偏好强上下文，title grounding 有歧义，能测试 free-form generation。

每个数据集按时间切分：train / validation / test。输入最近 \(L\) 个 item titles，输出 next title。ground truth 同时用 next-1 和 future-window next-m。

### 11.2 模型

Generative LLM baselines：

P5 / instruction-tuned recommendation。
TIGER / Semantic ID generative retrieval。
GenRec / direct title generation。
BIGRec / two-step grounding。
UGR，如果可复现，用作 uncertainty-aware generative baseline。

Uncertainty baselines：

Verbal self-confidence。
Length-normalized sequence probability。
Beam margin。
Semantic entropy。
Prompt perturbation disagreement。
WWW 2025 UQ-style decomposition。
GUIDER-style data/model uncertainty reranking。
ConfTuner-style verbal confidence tuning adapted to recommendation。

传统 recommender baselines：

SASRec、BERT4Rec、LightGCN、GRU4Rec 等作为 accuracy reference，不一定是主角。

### 11.3 RQ 设计

**RQ1: Do LLM recommenders know when their generated titles are correct?**
报告 VO-ECE、Brier、AURC、selective risk、CBU/WBC 四象限。

**RQ2: Is confidence entangled with item popularity?**
报告 Popularity-Confidence Slope、Tail Underconfidence Gap、head/mid/tail reliability diagrams、matched pair intervention。

**RQ3: Does confidence-guided exposure amplify echo chambers?**
做 offline replay + simulated feedback loop。报告 ExposureGini、TailCoverage、CategoryEntropy、user-interest calibration KL、多轮 confidence drift。

**RQ4: Can causal confidence calibration improve both accuracy and trust?**
比较 CURE-Rec 与 baselines 的 NDCG/Recall、ECE/Brier、tail coverage、hallucination rate、grounding error。

**RQ5: Can uncertainty decomposition prune noisy data without killing long-tail signal?**
不同噪声率、不同 pruning ratio 下，比较 NDCG/ECE/tail coverage。

### 11.4 必画图

1. **四象限图**：Correct/Incorrect × High/Low Confidence，按 item popularity 着色。
2. **Reliability diagram by popularity bucket**：head/mid/tail 三条线。
3. **Confidence-popularity partial regression plot**：控制 correctness 后的 residual confidence vs log popularity。
4. **Matched counterfactual pair plot**：同类同语义 head vs tail，confidence 差异。
5. **Feedback loop curve**：轮数增加时，mean confidence、true utility、tail coverage、exposure Gini 的变化。
6. **Pruning curve**：pruning ratio vs NDCG/ECE/tail coverage，对比 naive pruning 与 decomposed triage。
7. **Case study**：用户历史、小众 ground truth、LLM high-confidence popular wrong answer、CURE-Rec 如何降低 \(c_{\text{pop}}\)。

---

## 12. 与已有 uncertainty-aware LLM4Rec 的差异写法

Related work 里要非常明确地区分：

**WWW 2025 UQ for LLM-based Recommendation**：它证明 LLM recommendation uncertainty 可量化，并分解 recommendation uncertainty / prompt uncertainty，主要偏 measurement 和 uncertainty-aware prompting。你的工作进一步研究 confidence 与 popularity / exposure feedback 的 causal entanglement，并训练 generative recommender 输出 calibrated confidence。([openreview.net](https://openreview.net/forum?id=lhFcbb5q48))

**GUIDER**：它把 uncertainty 分成 data uncertainty 与 model uncertainty，用 logit evidence 在 single inference pass 中动态 reranking。你的工作不是只做 zero-shot reranking，而是做 generative title grounding、verbal confidence calibration、popularity deconfounding、training data triage 和 exposure-loop evaluation。([ojs.aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/38639/42601))

**UGR**：它已经提出 generative recommendation 的 uncertainty blindness，并用 uncertainty-aware rollout reward、difficulty-aware optimization、explicit confidence alignment 稳定训练。你的区别必须更尖锐：UGR 的重点是把 uncertainty 纳入 preference optimization；你的重点是 **causal confidence calibration under exposure**，尤其是 free-form title generation 中 confidence 被 popularity prior 污染后，会通过 exposure feedback 放大。([arxiv.org](https://arxiv.org/pdf/2602.11719))

**ConfTuner / ConfMAD / TrustGen**：这些是你灵感来源，但不是推荐方法。你借鉴 proper scoring、confidence expression、dynamic trust evaluation，提出 recommender-specific target：counterfactual user utility and exposure-aware calibration。([arxiv.org](https://arxiv.org/abs/2508.18847))

---

## 13. 论文贡献可以这样写

最终 paper contribution 建议写成四条：

**Contribution 1: A generative recommendation confidence benchmark.**
我们提出 title-level generative recommendation confidence evaluation，把 LLM 输出 title grounding 到 catalog，并通过 yes/no + confidence probe 系统衡量 correctness-confidence alignment、grounding overconfidence、popularity-confidence coupling、long-tail underconfidence、confidence-induced echo risk。

**Contribution 2: Empirical discovery of popularity-entangled confidence.**
我们发现 LLM recommenders 的 confidence 不只是 correctness signal，而显著受到 item popularity、model familiarity、catalog grounding ease 影响；这种 bias 导致 wrong-but-confident head recommendations 和 correct-but-uncertain tail recommendations。

**Contribution 3: CURE-Rec, a causal confidence calibration framework.**
我们提出把 confidence calibration target 从 \(P(correct|prediction)\) 改成 \(P(user accepts|do(exposure))\)，并通过 triangulated confidence elicitation、popularity residual disentanglement、RecBrier confidence tuning、risk-aware preference optimization 训练 generative recommender。

**Contribution 4: Uncertainty-guided data triage and exposure-safe inference.**
我们证明 uncertainty decomposition 能识别 noisy implicit-feedback samples，同时避免误删 long-tail hard positives；推理阶段通过 exposure-aware score 降低 echo risk，提高 calibration、tail coverage 和 robust utility。

---

## 14. 可能的 abstract 草稿

> Large language models are increasingly used as generative recommenders that directly produce item titles. Yet a fundamental question remains underexplored: when an LLM recommends an item, does it know whether the recommendation is correct? We show that confidence in LLM-based generative recommendation is not merely a reliability score, but an exposure-shaping variable: high-confidence items are more likely to be shown, clicked, and reinforced in future training data. Through a title-level generative recommendation benchmark, we reveal a systematic confidence-popularity coupling: LLM recommenders are overconfident on popular but incorrect items and underconfident on correct long-tail items, leading to confidence-amplified echo chambers. To address this, we propose CURE-Rec, a causal uncertainty-regularized framework that calibrates confidence to counterfactual user utility under exposure. CURE-Rec triangulates verbal, probabilistic, semantic, grounding, and counterfactual-stability signals; disentangles popularity-driven confidence; optimizes recommendation and confidence jointly with a recommendation-specific proper scoring objective; and uses decomposed uncertainty for noise-aware training data triage. Experiments on multiple generative recommendation benchmarks show that CURE-Rec improves not only recommendation accuracy, but also calibration, hallucination risk, long-tail coverage, and robustness to noisy implicit feedback.

---

## 15. 关键技术细节：如何让它不像“缝合”

顶会 reviewer 最怕看到“我们加了 uncertainty、加了 debias、加了 pruning、加了 reranking”。你必须用一个统一数学对象把所有模块串起来。这个统一对象就是：

\[
C(u,i) = P(R=1|u,i,do(E=1)).
\]

也就是 **exposure-counterfactual confidence**。

然后所有模块都服务于估计这个对象：

verbal confidence 是 noisy observation；
token likelihood 是 generation evidence；
semantic entropy 是 posterior dispersion；
grounding confidence 是 title-to-item uncertainty；
popularity residual 是 confounding correction；
pruning 是为了让训练数据更接近 counterfactual preference；
reranking 是用 calibrated confidence 控制 exposure。

这样就不是“把五个 trick 拼起来”，而是围绕同一个目标函数展开。

你可以在 introduction 里写：

> Existing confidence calibration assumes predictions are passively evaluated. In recommendation, predictions are actively exposed. Therefore, confidence is not only a property of a generated output, but also a cause of future observations.

这句话是全篇的理论锚点。

---

## 16. Reviewer 可能攻击点与防御

**攻击 1：推荐 ground truth 很稀疏，next item 不等于唯一正确答案。**
防御：使用 next-1、future-window、category/semantic graded relevance、explicit rating dataset、plausibility audit；calibration target 用 graded relevance 而不是只用 binary click。

**攻击 2：popularity 本身就是 relevance signal，你为什么要去掉？**
防御：不是去 popularity，而是去掉 “popularity-only confidence residual”。我们条件化 user mainstreamness、category、correctness，并保留 preference-supported popularity。

**攻击 3：echo chamber simulation 不真实。**
防御：使用多种用户 response model 做 sensitivity analysis；同时报告 one-step exposure concentration，不完全依赖 simulation；如果有真实曝光日志，用 IPS/DR 做更强验证。

**攻击 4：方法太复杂。**
防御：核心方法拆成三个必要模块，每个 ablation 对应一个 observation：去掉 grounding → hallucination confidence 上升；去掉 pop residual → head overconfidence 上升；去掉 RecBrier → verbal calibration 变差；去掉 data triage → noisy setting 变差。

**攻击 5：UGR 已经做了 uncertainty-aware generative recommendation。**
防御：UGR 主要解决 uncertainty blindness in preference optimization；你研究 confidence as exposure-causal variable，特别关注 title-level generation、popularity-confidence coupling、feedback-loop echo risk、uncertainty-based data triage。两者问题定义不同。

---

## 17. 最小可行版本与顶会完整版

最小可行版本不要一开始就做所有东西。建议分三阶段。

**Stage 1: Observation paper core**
先用 3 个数据集、3 类 LLM、5 类 confidence signal，证明 LLM4Rec 存在 popularity-entangled confidence、tail underconfidence、wrong-high-confidence popular hallucination。这个阶段就可能是一篇 strong workshop / findings 级工作。

**Stage 2: CURE-Rec method**
加入 RecBrier + popularity residual + risk-aware DPO/SFT。目标是证明 calibration、accuracy、tail coverage 同时改善。这是主会核心。

**Stage 3: Feedback and pruning extension**
做 echo simulation 与 uncertainty-guided data triage，作为顶会完整版的 additional contribution。如果篇幅不够，pruning 可以放 appendix 或作为第二篇 paper。

如果只投一篇顶会，我建议主线这样收束：

> Main paper: confidence-popularity coupling observation + causal confidence calibration framework + exposure-safe inference.
> Data pruning:作为 robustness application，不要喧宾夺主。

---

## 18. 最后的项目定位

这篇论文的真正卖点不是“uncertainty-aware LLM4Rec”，而是：

**LLM 推荐系统的 confidence 会塑造未来数据，因此 calibration 必须从静态正确性升级为 exposure-aware causal calibration。**

这个定位足够新，也足够大。它能连接 Bryan 组的 confidence calibration / trustworthiness 思路，但推荐场景给了它一个独特 twist：**confidence is performative**。LLM 在 QA 里说“我 90% 确定”通常只是报告信念；LLM 在推荐里说“我 90% 确定”会改变用户看到什么、点什么、系统未来学到什么。这个差异就是顶会级 originality 的来源。