# GMC-Dice-Loss
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![nnU-Net](https://img.shields.io/badge/nnU--Net-2.0-orange)](https://github.com/MIC-DKFZ/nnUNet)

**GraphCenter (GC)** for small bowel centerline extraction and **GraphMorph Dice (GMC-Dice) Loss** for improved small bowel volume segmentation.

---

## 📌 Features
- **Small bowel centerline extraction** from ground-truth segmentation volumes  
- **GMC-Dice Loss** for volumetric segmentation of tubular structures like the small bowel  
- Multiple **centerline-based Dice loss variations** to improve morphological accuracy  

---

## 🧬 Small Bowel Centerline Extraction
Extract small bowel centerlines from ground-truth binary segmentation volumes:

```bash
python ./gc_extracted 
```
seg_dir is the directory of the small bowel volume 
cen_dir is the directory of the output folder

## ⚙️ Installation

```bash
cd ./GMC-Dice-Loss
pip install -e .
```
## 🏋️ Training with GMC-Dice Loss

The dataset structure differs slightly from standard nnU-Net:
imagesTr/ — MRI images
labelsTr/ — 3 classes:
0 — background
1 — small bowel volume (ground truth)
2 — GC-extracted centerline
Preprocess the dataset:
```bash
nnUNetv2_plan_and_preprocess -d dataset_number --verify_dataset_integrity
```
Training commands:
```bash
nnUNetv2_train -d dataset_number 3d_fullres fold -tr nnUNetTrainerMatRecallPrecisionDSC ## GMC-Dice loss
nnUNetv2_train -d dataset_number 3d_fullres fold -tr nnUNetTrainer_CE_DC_CLDC ## Centerline Dice (Morphology Skeletonization) loss
nnUNetv2_train -d dataset_number 3d_fullres fold -tr -tr nnUNetTrainer_CE_DC_CLDC_topology ## Centerline Dice loss (Topology Skeletonization)
nnUNetv2_train -d dataset_number 3d_fullres fold -tr nnUNetTrainer_CE_DC_Skelite ## Centerline Dice loss (Skelite Skeletonization)
nnUNetv2_train -d dataset_number 3d_fullres fold -tr nnUNetTrainerSkeletonRecall ## Skeleton-Recall loss
nnUNetv2_train -d dataset_number 3d_fullres fold -tr nnUNetTrainerMatRecallPrecision ## GMC-Dice loss
```

## 📄 Citation
If you find this work useful, please cite our paper (TBD).
## 📞 Contact
For questions or collaboration, please contact us.