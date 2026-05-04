# Reference Baseline Audit

## Local Reference Files

- `references/README.md` exists and states that large PDFs, ZIP files, and extracted reference folders are local-only materials that should not be committed.
- `references/recprefer.zip` exists locally and contains `NH/` and `NR/` PDF materials.
- `references/NH/` and `references/NR/` exist locally as extracted PDF folders.
- `docs/related_work/` and `docs/related_work/baseline_notes.md` were not present during this audit.

No large reference PDFs, ZIPs, or extracted archives were added to version control.

## Recommended External Projects / Libraries

The local lightweight reference note does not name a specific runnable baseline library. The current task specification identifies RecBole as the preferred integration route for reviewer-grade SASRec and LightGCN baselines. Therefore, the implemented plan treats RecBole as the optional external baseline backend, while keeping all final metrics inside the TRUCE evaluator.

## Relevant Baselines

- SASRec is relevant as the standard self-attentive sequential recommendation baseline.
- LightGCN is relevant as the standard graph collaborative filtering baseline.
- BPR-MF or NeuMF may be useful later, but they were not prioritized because this stage requested SASRec and LightGCN first.
- Existing popularity, BM25/fallback, MF, and sequential Markov baselines remain useful comparators but do not replace SASRec or LightGCN.

## Feasible Now

- Exporting canonical TRUCE data to RecBole-compatible atomic files is feasible and implemented.
- Importing externally scored candidate predictions back into the TRUCE prediction schema is feasible and implemented.
- Writing RecBole config files for SASRec and LightGCN is feasible and implemented.
- Running RecBole training is not feasible in the current environment because `recbole` is not installed.

## Too Expensive or Incompatible Now

- Full SASRec and LightGCN training cannot be reported as paper-candidate evidence until the optional RecBole dependency is installed and the scoring/import path is executed.
- No external evaluator metrics should be copied into paper tables. External projects may train and score candidates, but TRUCE must compute Recall@10, NDCG@10, MRR@10, and HitRate@10.

## Integration Plan

1. Export each canonical processed dataset into RecBole atomic interaction/item/user files plus TRUCE candidate-set manifests.
2. Train SASRec or LightGCN through RecBole using fixed seeds and the same train/validation/test split.
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

RecBole is not installed in the current Python environment. The project now declares an optional `baselines` extra:

```bash
py -3 -m pip install -e .[baselines]
```

The external baseline scripts fail clearly when RecBole is missing instead of falling back to toy implementations.

## Paper-Candidate Evidence Status

- Paper-candidate evidence exists for existing TRUCE-evaluated baselines and CU-GR v2 artifacts already produced by the pipeline.
- SASRec and LightGCN are not paper-candidate evidence yet because training/scoring was not completed in this environment.
- The Amazon Video Games third-domain CU-GR v2 run is valid negative/diagnostic evidence, but it did not pass the held-out seed42 success gate.
