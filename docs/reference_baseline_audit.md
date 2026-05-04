# Reference Baseline Audit

## Local Reference Files

- `references/README.md` exists and states that large PDFs, ZIP files, and extracted reference folders are local-only materials that should not be committed.
- `references/recprefer.zip` exists locally and contains `NH/` and `NR/` PDF materials.
- `references/NH/` and `references/NR/` exist locally as extracted PDF folders.
- `docs/related_work/` and `docs/related_work/baseline_notes.md` were not present during this audit.

No large reference PDFs, ZIPs, or extracted archives were added to version control.

## Recommended External Projects / Libraries

The local lightweight reference note does not name a specific runnable baseline library. The current task specification identifies RecBole as the preferred integration route for reviewer-grade sequential and graph baselines. Therefore, the implemented plan treats RecBole as the optional external baseline backend, while keeping all final metrics inside the TRUCE evaluator.

## Relevant Baselines

- SASRec is relevant as the standard self-attentive sequential recommendation baseline.
- BERT4Rec and GRU4Rec are relevant as additional sequential recommendation baselines.
- LightGCN is relevant as the standard graph collaborative filtering baseline.
- BPR-MF or NeuMF may be useful later, but they were not prioritized because this stage requested RecBole-backed sequential/graph baselines first.
- Existing popularity, BM25/fallback, MF, and sequential Markov baselines remain useful comparators but do not replace RecBole-backed SASRec/BERT4Rec/GRU4Rec/LightGCN.

## Feasible Now

- Exporting canonical TRUCE data to RecBole-compatible atomic files is feasible and implemented.
- Importing externally scored candidate predictions back into the TRUCE prediction schema is feasible and implemented.
- Writing RecBole config files for SASRec, BERT4Rec, GRU4Rec, and LightGCN is feasible and implemented.
- Running RecBole training is feasible after installing the optional baseline stack. MovieLens 1M and Amazon Beauty completed for SASRec, BERT4Rec, GRU4Rec, and LightGCN and final metrics were computed by the TRUCE evaluator.

## Too Expensive or Incompatible Now

- Amazon Video Games RecBole baselines were not run in this stage because the minimum MovieLens/Beauty scope was completed and the third-domain CU-GR v2 gate is already a diagnostic non-pass. Do not tune or extend that domain unless explicitly requested.
- No external evaluator metrics should be copied into paper tables. External projects may train and score candidates, but TRUCE must compute Recall@10, NDCG@10, MRR@10, and HitRate@10.

## Integration Plan

1. Export each canonical processed dataset into RecBole atomic interaction/item/user files plus TRUCE candidate-set manifests.
2. Train the selected RecBole baseline through RecBole using fixed seeds and the same train/validation/test split.
3. Score exactly the TRUCE candidate set for each evaluation example.
4. Import per-candidate scores into `predictions.jsonl` using the unified TRUCE schema.
5. Run the TRUCE evaluator to compute final paper metrics.
6. Record source library, model name, training config, checkpoint path, seed, candidate protocol, and import time in prediction metadata.

## License / Provenance Notes

- No external project source code was copied into `src/`.
- The adapter references RecBole as an optional dependency through `pyproject.toml`; it does not vendor RecBole code.
- RecBole license and citation metadata should be verified from the installed package or upstream repository before final artifact packaging.
- Local reference PDFs and ZIP files remain provenance materials only and should not be committed.

## RecBole Availability

RecBole 1.2.1 was installed as an optional baseline dependency for this gate. The tested stack was Python 3.12.0, torch 2.10.0+cpu, RecBole 1.2.1, NumPy 1.26.4, SciPy 1.11.4, Ray 2.55.1, CPU only. RecBole 1.2.1 pins `ray<=2.6.3`, which does not have a Python 3.12 Windows wheel, so Python 3.10/3.11 is recommended for cleaner paper-grade reproduction.

The project declares an optional `baselines` extra:

```bash
py -3 -m pip install -e .[baselines]
```

The external baseline scripts fail clearly when RecBole is missing instead of falling back to toy implementations.

Environment check commands for this gate:

```powershell
py -3 -c "import sys; print(sys.version)"
py -3 -c "import torch; print(torch.__version__)"
py -3 -c "import recbole; print(recbole.__version__)"
```

Observed environment: Python 3.12.0, torch 2.10.0+cpu, RecBole 1.2.1, CPU only on Intel Core i5-1240P. CUDA was unavailable.

## Paper-Candidate Evidence Status

- Paper-candidate evidence exists for existing TRUCE-evaluated baselines and CU-GR v2 artifacts already produced by the pipeline.
- MovieLens 1M and Amazon Beauty SASRec/BERT4Rec/GRU4Rec/LightGCN now have TRUCE-evaluated prediction artifacts. Amazon Video Games RecBole baselines were not run and must not be marked passed.
- The Amazon Video Games third-domain CU-GR v2 run is valid negative/diagnostic evidence, but it did not pass the held-out seed42 success gate.
