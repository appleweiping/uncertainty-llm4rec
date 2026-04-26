# Day35 Books/Electronics medium_5neg Cold-Rate Report

This report applies the same diagnostic used for Movies. It separates candidate count from cold-candidate composition and reports cold rate under train_candidate_vocab, train_history_vocab, and train_backbone_vocab.

Using train_backbone_vocab:

- books: valid pos/neg/all cold `0.8765` / `0.9487` / `0.9367`; test pos/neg/all cold `0.8750` / `0.9513` / `0.9386`.
- electronics: valid pos/neg/all cold `0.7455` / `0.9385` / `0.9063`; test pos/neg/all cold `0.7640` / `0.9378` / `0.9088`.

Interpretation: if positive cold is low and warm-strict users are available, the domain is more suitable for ID-based backbone evaluation. If all-candidate cold is high, content-carrier cold diagnostics are safer.
