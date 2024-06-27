# BackMix: Mitigating Shortcut Learning in Echocardiography with Minimal Supervision

Accepted at MICCAI 2024: Pre-print available soon

![](C:\Users\KitBransby\GitHub\BackMix\evaluation\figures\schematic.png)

### Requirements

Set up environment and install dependencies

```
# Create a virtual environment (recommended)
conda create -n backmix --file requirements.txt
```

### Dataset

We use two datasets in this paper. 
Access to the TMED2 dataset can be obtained [here](https://tmed.cs.tufts.edu/index.html). WASE-Normals is a proprietary dataset which is not available. 

For preprocessing steps, please refer to our paper (Section 3.1). We assume all data stored in a separate data directory set as the `DATA_ROOT` with the following structure:
```
 Data
   ├── Dataset 1 (e.g TMED)
   │   ├── train
   │   │   ├── 0001_Y.npy
   │   │   ├── 0002_Y.npy
   │   │   └── ...
   │   ├── val 
   │   ├── test
   │   └── masking
   │        ├── train
   │        │  ├── 0001_Y.npy
   │        │  ├── 0002_Y.npy
   │        │  └── ...
   │        ├── val 
   │        └── test
   └── Dataset 2
```
Where Y is the classification label. To use your own data, just follow this structure.

Binary masks for the area of focus (e.g the Ultrasound sector) are required to train and evaluate BackMix, although this can be reduced to ~5-10% of data for training using wBackMix. 

### Train 

We train several models in this work. Instructions for each are below. 

```
# Train the baseline:
python train_vc.py --CONFIG avc_tmed2_resnet --DATA_ROOT <path-to-data-folder>

# Train w/ BackMix
python train_vc.py --CONFIG avc_tmed2_resnet_random_bg --DATA_ROOT <path-to-data-folder>

# Train w/ BackMix, but only apply aug to a fraction f of training examples (semi-supervised)
python train_vc.py --CONFIG avc_tmed_resnet_random_bg_semi_${f} --DATA_ROOT <path-to-data-folder>

# As above, but with wBackMix (lambda=1, f=0.05)
python train_vc_weighted.py --CONFIG avc_tmed_resnet_random_bg_semi_0_05_weighted_1 --DATA_ROOT <path-to-data-folder>
```

Results are saved to ``results/<RUN_ID>`` whenever a training run is launched. 

### Evaluate 

To evaluate on TMED 
```
python evaluate_vc.py --CONFIG avc_tmed_2 --RUN_ID {specify results folder which contains weights}
```

Evaluate results are saved to `results/<RUN_ID>`

### Using your own data. 

To train and evaluate on your own dataset you will need to make some changes to the code:
* Save dataset using the expected structure (see above) and save to your `DATA_ROOT` folder.
* Create a `.yaml` config file in `config/` for the dataset. 
* Register the dataset in `utils/load_dataset.py` using the same `DATASET` name as in the config
* Simply train and evaluating using the commands above. 

