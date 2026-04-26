# Day36 Small-Domain Cold-Rate Report

Cold rate is computed with train_candidate_vocab, train_history_vocab, and train_backbone_vocab. The route decision uses train_backbone_vocab.

- books: valid pos/neg/all cold `0.0540` / `0.0188` / `0.0247`; test pos/neg/all cold `0.0440` / `0.0216` / `0.0253`.
- electronics: valid pos/neg/all cold `0.0520` / `0.0276` / `0.0317`; test pos/neg/all cold `0.0600` / `0.0220` / `0.0283`.
- movies: valid pos/neg/all cold `0.0280` / `0.0160` / `0.0180`; test pos/neg/all cold `0.0280` / `0.0120` / `0.0147`.

If small-domain cold rates are low, they can support ID-based backbone sanity. If they are still cold, they should be used for calibration sanity or content-carrier checks only.
