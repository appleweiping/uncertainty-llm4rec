# Framework-Day3 LoRA Training Plan

1. Reference DecodingMatters for a minimal train/eval entry point and dataset-to-causal-LM wrapping.
2. Implement our own Dataset class because `data_done_lora` uses closed-candidate JSONL and must preserve CEP metadata/fallback flags.
3. First train Qwen-LoRA as a recommendation baseline, not as the CEP method.
4. After a trained baseline exists, compare:
   - Qwen-LoRA baseline
   - Qwen-LoRA + CEP calibrated posterior
   - Qwen-LoRA + evidence risk
   - Qwen-LoRA + full CEP framework
5. Before training, Framework-Day4 should run tokenizer/parser/inference smoke and verify the server-side Qwen3-8B path.
