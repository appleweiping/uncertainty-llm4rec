# Week7.7 Teacher Reliability Summary

These pointwise uncertainty diagnostics provide the calibration/reliability evidence for the SRPD teacher layer.

|Domain|Samples|Accuracy|AvgConf|Brier|ECE|AUROC|SRPD Role|
|---|---:|---:|---:|---:|---:|---:|---|
|beauty|5838|0.656560|0.749455|0.264322|0.218777|0.618473|positive-domain distillation evidence|
|books|3000|0.836333|0.791580|0.137134|0.044753|0.667293|positive-domain distillation evidence|
|electronics|3000|0.762333|0.730627|0.239543|0.187660|0.610643|repair-domain evidence; direct-anchor worked better than preference v6 so far|
|movies|3000|0.738333|0.550613|0.297879|0.261853|0.433191|failure-boundary evidence; low teacher reliability / weak AUROC warning|
