# Week7.7 Four-Domain SRPD Full-Metric Matrix

This table extends the utility-only matrix with coverage, exposure, parsing, and out-of-candidate stability metrics from local server-synced files.

|Domain|Scope|Method|Winner|NDCG@10|MRR|Coverage@10|HeadExposure@10|LongTail@10|Parse|OOC|Role|
|---|---|---|---|---|---|---|---|---|---|---|---|
|beauty|full973|direct|no|0.614031|0.489999|1.004263|0.216577|1.000000|0.994861|0.005139|non_uncertainty decision baseline|
|beauty|full973|structured_risk|no|0.614078|0.490048|1.000000|0.214457|1.000000|0.994861|0.005139|hand-crafted uncertainty-aware ranking baseline|
|beauty|full973|SRPD-v1|no|0.626119|0.506458|1.000000|0.214457|1.000000|1.000000|0.000000|plain structured-risk teacher SFT ablation|
|beauty|full973|SRPD-v2|yes|0.635366|0.518448|1.000853|0.214457|1.000000|0.998972|0.001028|uncertainty-weighted trainable distillation|
|beauty|full973|SRPD-v3|no|0.623807|0.503357|1.000853|0.214494|1.000000|0.998972|0.001028|pairwise-support retained ablation|
|beauty|full973|SRPD-v4|no|0.625777|0.506132|1.000000|0.214494|1.000000|1.000000|0.000000|gap-gated repair ablation against v2|
|beauty|full973|SRPD-v5|no|0.624992|0.505190|1.000000|0.214322|1.000000|1.000000|0.000000|direct-anchor repair ablation against v4|
|books|small500|direct|no|0.639827|0.523900|0.998999|0.206344|1.000000|0.998000|0.002000|non_uncertainty decision baseline|
|books|small500|structured_risk|no|0.639514|0.523500|1.000000|0.206000|1.000000|0.998000|0.002000|hand-crafted uncertainty-aware ranking baseline|
|books|small500|SRPD-v2|yes|0.705707|0.611767|1.000000|0.206000|0.998726|0.998000|0.002000|uncertainty-weighted trainable distillation|
|books|small500|SRPD-v4|no|0.698716|0.602467|0.999500|0.206137|0.998726|1.000000|0.000000|gap-gated repair ablation against v2|
|books|small500|SRPD-v5|no|0.697400|0.600800|0.998499|0.206275|0.996178|1.000000|0.000000|direct-anchor repair ablation against v4|
|electronics|small500|direct|no|0.658301|0.547167|1.000000|0.228667|1.000000|1.000000|0.000000|non_uncertainty decision baseline|
|electronics|small500|structured_risk|no|0.658301|0.547167|1.000000|0.228667|1.000000|1.000000|0.000000|hand-crafted uncertainty-aware ranking baseline|
|electronics|small500|SRPD-v2|no|0.642958|0.527833|0.946573|0.233358|0.933518|0.998000|0.002000|uncertainty-weighted trainable distillation|
|electronics|small500|SRPD-v4|no|0.659405|0.549233|1.000000|0.228667|1.000000|1.000000|0.000000|gap-gated repair ablation against v2|
|electronics|small500|SRPD-v5|yes|0.662100|0.552833|1.000000|0.228409|1.000000|1.000000|0.000000|direct-anchor repair ablation against v4|
|movies|small500|direct|no|0.573060|0.438967|1.001035|0.209279|0.997426|0.992000|0.008000|non_uncertainty decision baseline|
|movies|small500|structured_risk|yes|0.573183|0.439100|1.000000|0.209000|1.000000|0.992000|0.008000|hand-crafted uncertainty-aware ranking baseline|
|movies|small500|SRPD-v2|no|0.537348|0.392233|0.999483|0.209209|0.998713|0.998000|0.002000|uncertainty-weighted trainable distillation|
|movies|small500|SRPD-v4|no|0.546389|0.404367|1.000000|0.209000|1.000000|1.000000|0.000000|gap-gated repair ablation against v2|
|movies|small500|SRPD-v5|no|0.543351|0.400200|1.000000|0.209000|1.000000|1.000000|0.000000|direct-anchor repair ablation against v4|
