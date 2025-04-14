## Model loader
1. Need to adjust your weight mapping yaml according to your training script, including model sharding parameters (eg. tensor parallel, pipeline parallel), all set to 1 to get the whole model on one device.
