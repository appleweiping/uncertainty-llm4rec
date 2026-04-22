# Week7.7 Four-Domain SRPD Final Matrix

Generated from server-synced result files under `server_sync/week7_20260421/week7_7_four_domain_final`.

| Domain | Scope | Winner | Winner NDCG@10 | Winner MRR | Interpretation |
|---|---:|---|---:|---:|---|
| beauty | full973 | SRPD-v2 | 0.635366 | 0.518448 | Positive-domain winner remains SRPD-v2. |
| books | small500 | SRPD-v2 | 0.705707 | 0.611767 | Positive-domain winner remains SRPD-v2. |
| electronics | small500 | SRPD-v5 | 0.662100 | 0.552833 | Direct-anchor repair SRPD-v5 is strongest. |
| movies | small500 | structured_risk | 0.573183 | 0.439100 | Structured risk/direct line remains strongest; SRPD-v4 is best SRPD repair. |

## Full Rows

| Domain | Method | NDCG@10 | MRR | Samples | Winner |
|---|---|---:|---:|---:|---|
| beauty | direct | 0.614031 | 0.489999 | 973 | no |
| beauty | structured_risk | 0.614078 | 0.490048 | 973 | no |
| beauty | SRPD-v1 | 0.626119 | 0.506458 | 973 | no |
| beauty | SRPD-v2 | 0.635366 | 0.518448 | 973 | yes |
| beauty | SRPD-v3 | 0.623807 | 0.503357 | 973 | no |
| beauty | SRPD-v4 | 0.625777 | 0.506132 | 973 | no |
| beauty | SRPD-v5 | 0.624992 | 0.505190 | 973 | no |
| books | direct | 0.639827 | 0.523900 | 500 | no |
| books | structured_risk | 0.639514 | 0.523500 | 500 | no |
| books | SRPD-v2 | 0.705707 | 0.611767 | 500 | yes |
| books | SRPD-v4 | 0.698716 | 0.602467 | 500 | no |
| books | SRPD-v5 | 0.697400 | 0.600800 | 500 | no |
| electronics | direct | 0.658301 | 0.547167 | 500 | no |
| electronics | structured_risk | 0.658301 | 0.547167 | 500 | no |
| electronics | SRPD-v2 | 0.642958 | 0.527833 | 500 | no |
| electronics | SRPD-v4 | 0.659405 | 0.549233 | 500 | no |
| electronics | SRPD-v5 | 0.662100 | 0.552833 | 500 | yes |
| movies | direct | 0.573060 | 0.438967 | 500 | no |
| movies | structured_risk | 0.573183 | 0.439100 | 500 | yes |
| movies | SRPD-v2 | 0.537348 | 0.392233 | 500 | no |
| movies | SRPD-v4 | 0.546389 | 0.404367 | 500 | no |
| movies | SRPD-v5 | 0.543351 | 0.400200 | 500 | no |
