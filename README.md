# Solution of Team zzm3zz for FLARE23 Challenge
**A Semi-Supervised Abdominal Multi-Organ Pan-Cancer Segmentation Framework with Knowledge Distillation and Multi-Label Fusion** \
*Zengmin Zhang Xiaomeng Duan Yanjun Peng Zhengyu Li* \

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the solution of team zzm3zz for [MICCAI FLARE23](https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details-overview) Challenge. The details of our method are described in our [paper](https://openreview.net/forum?id=PLFBzKnjOt). 

You can reproduce our method as follows step by step:

## 0. Grouping Partial Label Data
### 0.1 Get Tumor sub-dataset

## 1. Training Attention nnUNet for Tumor Pseudo-Labels
### 1.1. Prepare partial Labeled Data of Tumors
Following nnUNet, give a TaskID (e.g. Task022) to the 50 labeled data and organize them folowing the requirement of nnUNet.

    nnUNet_raw_data_base/nnUNet_raw_data/Task009_FLARE22_Tumor/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
### 1.2. Conduct automatic preprocessing using nnUNet.
Here we do not use the default setting.
```
nnUNet_plan_and_preprocess -t 9 -pl2d ExperimentPlanner2D_Attention -pl3d None
```
### 1.3. Training Teacher Model Attention nnUNet by all fold 
```
nnUNet_train 2d nnUNetTrainerV2_Attention 9 all -p nnUNetPlansAttention
```
### 1.5. Generate Pseudo Labels for missing tumor annotations Data
```
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER  -t 9  -tr nnUNetTrainerV2_Attention  -m 2d  -p nnUNetPlansAttention  --disable_tta 
```

## 2. Mutil-Label_Fusion
### 2.1 Fuse partial labels and tumor pseudo-labels
```
run part_tumor_organ_fuse.py
```
### 2.2 Fuse organs pseudo-labels
```
run tumor_organs_fuse.py
```

## 3. Train Student Model Small nnUNet 
### 3.1. Conduct automatic preprocessing using nnUNet
Here we use the plan designed for small nnUNet.
```
nnUNet_plan_and_preprocess -t 23 -pl3d ExperimentPlanner3D_FLARE22Small -pl2d None
```
### 3.2. Train small nnUNet on all training data
```
nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Small 23 all -p nnUNetPlansFLARE22Small
```

## 4. Do Efficient Inference with Small nnUNet
We modify a lot of parts of nnunet source code for efficiency. Please make sure the code backup is done and then copy the whole repo to your nnunet environment.
```
nnUNet_predict -i INPUT_FOLDER  -o OUTPUT_FOLDER  -t 23  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
 -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta
```


