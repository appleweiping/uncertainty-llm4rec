# Framework-Day6 Beauty Listwise Eval Report

## Scope

This is the server-side adapter inference / parsing / ranking smoke for the Day6 Beauty listwise Qwen-LoRA baseline. It does not use confidence, evidence, CEP fusion, or API calls.

## Parser Result

- parse success rate: `0.8672`
- schema valid rate: `0.8516`
- invalid item rate: `0`
- duplicate item rate: `0`

## Ranking Metrics on 128 Samples

| method | NDCG@10 | MRR | HR@1 | HR@3 | NDCG@3 | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-LoRA Day6 small | 0.5683 | 0.4158 | 0.2422 | 0.4531 | 0.3610 | 0.4823 |
| Random | 0.5505 | 0.4076 | 0.1563 | 0.5156 | 0.3615 | 0.4892 |

HR@10 is trivial for the Beauty 5neg candidate pool because each user has six candidates. It is not used as a main claim.

## Interpretation

The train/infer/parse/evaluate loop works. Performance is not yet strong: LoRA is only slightly above random on NDCG@10/MRR, while HR@3 and NDCG@5 are lower than random. The next step is Day7 parser and evaluation diagnosis, not larger training or CEP integration.
