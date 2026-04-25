# Backbone Calibration Paper Snippet

Although sequential recommenders provide useful ranking scores, these scores should not be interpreted as calibrated probabilities. In our diagnostic, SASRec-style, GRU4Rec, and Bert4Rec candidate scores remain miscalibrated under naive probability mappings such as sigmoid, min-max normalization, and per-user softmax. This does not indicate that the backbones fail as recommenders; rather, it reflects that ranking logits and sequence scores are optimized for ordering candidates, not for estimating calibrated relevance probabilities.

This distinction motivates separating ranking ability from uncertainty estimation. External backbones provide candidate ranking signals, while CEP provides an evidence-grounded calibrated relevance posterior and an auxiliary evidence-risk signal. In downstream plug-in experiments, CEP is therefore used to complement backbone ranking scores rather than replacing the backbone or treating raw model scores as confidence.
