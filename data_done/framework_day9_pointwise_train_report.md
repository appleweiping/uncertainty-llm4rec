# Framework-Day9 Pointwise-v1 Qwen-LoRA Train Report

## Scope

This trains the Beauty pointwise relevance baseline only, later aggregated into a candidate ranking. It uses raw relevance labels, not calibrated probabilities and not CEP.

## Status

- Status: `success`
- Blocked reasons: `none`
- OOM status: `False`
- Ready for Day6 real train: `True`

## Config

- train samples: `2000`
- eval samples: `128`
- max steps: `300`
- batch size: `1`
- gradient accumulation steps: `4`
- max seq len: `1536`
- LoRA rank/alpha/dropout: `8` / `16` / `0.05`
- adapter output dir: `artifacts/lora/qwen3_8b_beauty_pointwise_day9_small`

## Loss

- loss first: `4.8161444664001465`
- loss last: `0.020075034350156784`
- loss NaN count: `0`
- recorded losses: `[4.8161444664001465, 4.028693199157715, 2.8262252807617188, 2.006789207458496, 1.5109670162200928, 1.1777830123901367, 0.6651659607887268, 0.2576904594898224, 0.02883383259177208, 0.01263604685664177]`

## GPU

- peak GPU memory GB: `19.0893`

## Interpretation

Passing this tiny train means the Qwen3-8B LoRA baseline infrastructure can perform optimizer steps on server data. It is not a performance result and should not be used as CEP/framework evidence.
