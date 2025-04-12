## Model loader
1. Need to adjust your weight mapping yaml according to your training script, including model sharding parameters (eg. tensor parallel, pipeline parallel)
2. If **dist_ckpt_parallel_save_within_dp** in your yaml was set to **True** when training, the number of process used to load the model to cpu will be set to $TP \times PP \times DP$.
