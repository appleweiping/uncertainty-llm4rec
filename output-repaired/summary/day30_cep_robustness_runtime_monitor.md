# Day30 CEP Robustness Runtime Monitor

Last update: 2026-04-26 05:38:48

| noise_type | noise_level | expected_rows | current_rows | rolling_parse_success | status | resume_command |
|---|---:|---:|---:|---:|---|---|
| history_dropout | 0.1 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_dropout_0.1.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| history_dropout | 0.2 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_dropout_0.2.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| history_dropout | 0.3 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_dropout_0.3.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| candidate_text_dropout | 0.1 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_candidate_text_dropout_0.1.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| candidate_text_dropout | 0.2 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_candidate_text_dropout_0.2.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| candidate_text_dropout | 0.3 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_candidate_text_dropout_0.3.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| history_swap_noise | 0.1 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_swap_noise_0.1.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| history_swap_noise | 0.2 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_swap_noise_0.2.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| history_swap_noise | 0.3 | 3000 | 3000 | 1.0000 | complete | `py -3.12 main_infer.py --config configs\exp\beauty_robustness_500_history_swap_noise_0.3.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
