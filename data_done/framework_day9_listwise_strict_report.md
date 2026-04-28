# Framework-Day9 Listwise-v2 Strict Qwen-LoRA Train Report

## Scope

This trains the Beauty listwise-v2 strict prompt baseline only. It validates whether matching strict train/inference formulation improves the local Qwen-LoRA recommender. It does not implement confidence, evidence, calibrated posterior, or CEP fusion.

## Status

- Status: `success`
- Blocked reasons: `none`
- OOM status: `False`
- Ready for Day6 real train: `True`

## Config

- train samples: `622`
- eval samples: `128`
- max steps: `300`
- batch size: `1`
- gradient accumulation steps: `4`
- max seq len: `2048`
- LoRA rank/alpha/dropout: `8` / `16` / `0.05`
- adapter output dir: `artifacts/lora/qwen3_8b_beauty_listwise_strict_day9_small`

## Loss

- loss first: `0.3669952154159546`
- loss last: `0.00704180309548974`
- loss NaN count: `0`
- recorded losses: `[0.3669952154159546, 0.2932848036289215, 0.5305477976799011, 0.2239743322134018, 0.11837977170944214, 0.007280443329364061, 0.055173248052597046, 0.010000179521739483, 0.04099920019507408, 0.01598360762000084]`

## GPU

- peak GPU memory GB: `20.2553`

## Interpretation

Passing this tiny train means the Qwen3-8B LoRA baseline infrastructure can perform optimizer steps on server data. It is not a performance result and should not be used as CEP/framework evidence.
