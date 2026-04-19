# Week7 Server Execution

The intended Week7 execution loop is:

1. Edit code and configs on the local workstation.
2. Commit and push the branch to GitHub.
3. Log in to the server and enter the project checkout.
4. Pull the latest branch with `git pull --ff-only`.
5. Activate the conda environment that owns PyTorch, Transformers, Accelerate, and related inference dependencies.
6. Run `main_backend_check.py` or `scripts/week7_day1_server_backend_check.sh` before batch execution.
7. Launch the batch registry with `main_batch_run.py --batch_config configs/batch/week7_local_scale.yaml --run`.
8. Inspect `outputs/summary/week7_day2_batch_status.csv` and logs under `outputs/logs/batch/`.

For the current local-HF path, avoid editing repository files just to switch Python environments. Prefer:

```bash
PYTHON_BIN=/path/to/env/bin/python python main_batch_run.py \
  --batch_config configs/batch/week7_local_scale.yaml \
  --run
```

or use the helper script:

```bash
PYTHON_BIN=/path/to/env/bin/python BATCH_CONFIG=configs/batch/week7_local_scale.yaml \
  bash scripts/run_sync_and_batch.sh "$PWD"
```

Credentials, temporary SSH links, remote desktop passwords, and one-off tokens must stay outside this repository. If a server path changes, only update model paths and non-sensitive workflow notes.

