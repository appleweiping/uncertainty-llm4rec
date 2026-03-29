import json
from pathlib import Path
from src.data.noise import apply_noise_to_sample

input_path = "data/processed/test.jsonl"
output_path = "data/processed/test_noisy.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

noisy_data = [apply_noise_to_sample(x) for x in data]

Path(output_path).parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for item in noisy_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Saved noisy dataset to:", output_path)