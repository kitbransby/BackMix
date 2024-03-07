# BackMix
Under review at MICCAI 2024


### Dataset

Access to the TMED dataset can be obtained [here](https://tmed.cs.tufts.edu/index.html). 

### Train 

To train the baseline:
`python train_vc.py --CONFIG avc_tmed2_resnet`

To train w/ BackMix
`python train_vc.py --CONFIG avc_tmed2_resnet_random_bg`

To train w/ BackMix, but only apply aug to a fraction f of training examples (semi-supervised)
`python train_vc.py --CONFIG avc_tmed_resnet_random_bg_semi_${f}`

To train as above, but with wBackMix (lambda=1, f=0.05)
`python train_vc_weighted.py --CONFIG avc_tmed_resnet_random_bg_semi_0_05_weighted_1`

All training files require a random seed using the `--SEED` arg. 

### Evaluate 

To evaluate on TMED (regardless of config used for training)
`python evaluate_vc --CONFIG avc_tmed_2 --RUN_ID {specify results folder which contains weights}`

### Weights 

Weights are provided in `weights/`. Each model is run three times with a different seed (5, 10, 15). 

