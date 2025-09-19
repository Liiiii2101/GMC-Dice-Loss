"""
Assumes folder structure
someModel/
---Train/
-----seg/
-------pt_001.nii
-----sdf/
-------pt_001.nii
---Val/
-----seg/
-------pt_010.nii
-----sdf/
-------pt_010.nii
---Test/
-----seg/
-------pt_021.nii
-----sdf/
-------pt_021.nii
"""
import argparse
import os
import json

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, PrecisionRecallDisplay
from tqdm import tqdm
import sys
#from UNet.model.pl_wrappers import DiceLoss
from utils.datasets import get_sdf
from utils.utils import torch_from_nii
from torchmetrics.classification import Dice
from skimage.morphology import skeletonize, dilation
from metrics.cldice import clDice,clDice_mat
from metrics.BettiMatching import *
from metrics.betti_errors import BettiNumberMetric
sys.path.append('/home/l.cai/Betti-Matching-3D/build')
import betti_matching
from metrics.betti_losses import FastBettiMatchingLoss
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import gudhi

def calculate_betti_numbers(volume):
    """
    Function to calculate the 0th and 1st Betti numbers of a 3D volume using persistent homology.
    """
    # Ensure the input is a binary volume
    volume = np.array(volume, dtype=bool)

    # Step 1: Compute the connected components (β₀)
    # Use `measure.label` to find connected components in the 3D binary volume
    labeled_volume, num_features = ndi.label(volume)
    
    # β₀ is the number of connected components
    beta_0 = num_features

    # Step 2: Compute β₁ (the number of holes or loops)
    # Use Gudhi (a persistent homology library) to calculate persistent homology
    # Convert the 3D volume into a simplicial complex using Gudhi's RipsComplex
    rips_complex = gudhi.RipsComplex(points=np.argwhere(volume), max_edge_length=2)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Compute the persistent homology
    persistence = simplex_tree.persistence()
    
    # β₁ corresponds to the number of loops or 1-dimensional voids (holes) in the structure
    beta_1 = sum(1 for p in persistence if p[0] == 1 and p[1] is not None)  # count 1-dimensional features

    return beta_0, beta_1


def betti_error(ground_truth, predicted):
    """
    Compute the absolute Betti number errors between ground truth and predicted volumes.
    """
    beta_0_gt, beta_1_gt = calculate_betti_numbers(ground_truth)
    beta_0_pred, beta_1_pred = calculate_betti_numbers(predicted)

    # Absolute errors for β₀ and β₁
    betti_error_0 = abs(beta_0_gt - beta_0_pred)
    betti_error_1 = abs(beta_1_gt - beta_1_pred)

    return betti_error_0, betti_error_1


# Example: Compute Betti errors for two sample volumes
ground_truth_volume = np.random.random((50, 50, 50)) > 0.5  # Random binary volume (GT)
predicted_volume = np.random.random((50, 50, 50)) > 0.5  # Random binary volume (Pred)

betti_errors = betti_error(ground_truth_volume, predicted_volume)
print("Betti Error (β₀):", betti_errors[0])
print("Betti Error (β₁):", betti_errors[1])

betti_number_metric = BettiNumberMetric(
        num_processes=16,
        ignore_background=True,
        eight_connectivity=True#eight_connectivity
    )
BM_loss = FastBettiMatchingLoss()

def compute_metrics(t, relative=False, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(t[0], t[1], relative=relative, comparison=comparison, filtration=filtration, construction=construction)
    return BM.loss(dimensions=[0,1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(threshold=0.5, dimensions=[0,1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(threshold=0.5, dimensions=[1])


def calculate_tubed_skeleton(seg_all, do_tube):
    seg_all = seg_all.numpy()
    bin_seg = (seg_all > 0)
    seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
    
    # Skeletonize
    if not np.sum(bin_seg) == 0:
        skel = skeletonize(bin_seg)
        skel = (skel > 0).astype(np.int16)
        if do_tube:
            skel = dilation(skel)
        skel *= seg_all.astype(np.int16)
    
    return torch.from_numpy(skel)


#Hausdorff = HausdorffDistance()
dice_metric = Dice(ignore_index=0, average='micro')
#clDice_metric = ClDiceMetric(ignore_background=False)

def sdf2isdf(sdf):
    return torch.abs(sdf - sdf.max())


def score(pred_seg, pred_sdf, pred_seg_pcr, gt_seg_pcr, gt_seg, gt_sdf=None):
    # Calculate DICE
    dice = dice_metric(pred_seg, gt_seg.int())
    cldice = clDice(pred_seg.numpy(), gt_seg.numpy())#dice_metric(calculate_tubed_skeleton(pred_seg, True), calculate_tubed_skeleton(gt_seg, True).int())
    cldice_pcr = clDice_mat(pred_seg.numpy(), gt_seg.numpy(), pred_seg_pcr.numpy(), gt_seg_pcr.numpy())
    print(cldice_pcr)
    results = betti_matching.compute_matching(
    pred_seg.numpy(), gt_seg.numpy()#,
    #return_target_unmatched_pairs=True#np.zeros((3, 4,5)), np.zeros((3, 4,5)), 
    #include_input1_unmatched_pairs=True, 
    #include_input2_unmatched_pairs=True
)
    #print(dir(results), results.num_matched, results.num_unmatched_input1, results.num_unmatched_input2)
    # Ensure `results[0]` is a BettiMatchingResult object and access its attributes directly
    b0_pred = results.num_matched[0] + results.num_unmatched_input1[0] + 1
    #b0_pred = results[0].num_matches_by_dim[0] + results[0].num_unmatched_prediction_by_dim[0] + 1
    b1_pred = results.num_matched[1] + results.num_unmatched_input1[1]
    b0_label = results.num_matched[0] + results.num_unmatched_input2[0] + 1
    b1_label = results.num_matched[1] + results.num_unmatched_input2[1]
    #print('b0_pred' , b0_pred, b1_pred, b0_label, b1_label)

    #b0_errors[i, c - 1] = abs(b0_pred - b0_label)
    #b1_errors[i, c - 1] = abs(b1_pred - b1_label)
    b0_errors = abs(b0_pred - b0_label)
    b1_errors = abs(b1_pred - b1_label)

    #print(type(pred_seg), len(results.input2_matched_death_coordinates),np.array(results.input2_matched_death_coordinates).strides[-1])#.strides[-1])

    # Compute betti matching error
    #bm_losses = BM_loss._betti_matching_loss(pred_seg, gt_seg, results)

    # Compute normalized betti matching error
    #normalized_bm_losses[i, c - 1] = min(1, (bm_losses[i, c - 1]) / (b0_label + b1_label))
    #b0_errors, b1_errors = betti_error(pred_seg.numpy(), gt_seg.numpy())
    print('errors', b0_errors, b1_errors)
    # betti_errors = betti_error(gt_seg, pred_seg)
    # print("Betti Error (β₀):", betti_errors[0])
    # print("Betti Error (β₁):", betti_errors[1])
    #results = betti_matching.compute_matching(pred_seg.unsqueeze(0).unsqueeze(0).numpy(), gt_seg.unsqueeze(0).unsqueeze(0).numpy(), return_target_unmatched_pairs=False)
    #betti_number_metric(y_pred=pred_seg.unsqueeze(0), y=gt_seg.unsqueeze(0))
    #b0, b1, bm0, bm1, bm, norm_bm = betti_number_metric.aggregate()
    #Betti_matching_error, Betti_matching_error_0, Betti_matching_error_1, Betti_error, Betti_0_err, Betti_1_err = compute_metrics([pred_seg, gt_seg],  relative=True, comparison='union', filtration='superlevel', construction='V')
    #b0, b1 = calculate_betti_errors(pred_seg, gt_seg)
    #test_labels_one_hot = torch.nn.functional.one_hot(gt_seg.long(), num_classes=2)  # Shape: (batch, height, width, length, 2)
    #test_labels_one_hot = test_labels_one_hot.permute(0, 4, 1, 2, 3)  # Now (batch, 2, height, width, length)

    # Reshape `test_outputs` into (batch, 2, height, width, length)
    # If the model output is (batch, 1, height, width, length), duplicate the single channel for binary segmentation
    # if test_outputs.shape[1] == 1:
    #     test_outputs = pred_seg.repeat(1, 2, 1, 1, 1)  # Duplicates the channel for binary classification

    # # Get the class index with the highest value for each voxel (for binary classification)
    # pred_indices = torch.argmax(test_outputs, dim=1)  # Shape
    # pred_indices = pred_indices.unsqueeze(1)

   
    # betti_number_metric(y_pred=pred_indices, y=test_labels)
    # b0, b1, bm0, bm1, bm, norm_bm = betti_number_metric.aggregate()
    # print(b0, b1, bm0, bm1, bm, norm_bm)
    #pred_indices = torch.argmax(pred_seg, dim=1)
    #one_hot_pred = (torch.nn.functional.one_hot(pred_indices, num_classes=pred_seg.shape[1])).unsqueeze(0)
    #one_hot_pred = one_hot_pred.permute(0, 3, 1, 2)
    #print(one_hot_pred.shape, gt_seg.unsqueeze(0).shape)
    #clDice_metric(y_pred=one_hot_pred, y=gt_seg)
    print(dice.item(), cldice)
    

    # Calculate Hausdorff proxy
    pred_seg = (pred_seg >= 0.5).int()  # convert probabilities to binary
    sdf_temp = get_sdf(pred_seg[None, ...])[0]
    gt_sdf = get_sdf(gt_seg[None, ...])[0]
    pred2gt = sdf_temp[gt_seg.bool()].max()
    gt2pred = gt_sdf[pred_seg.bool()].max()
    hausdorff = max(pred2gt, gt2pred)

    precision, recall, _, _ = precision_recall_fscore_support(gt_seg.bool().numpy().flatten(), pred_seg.bool().numpy().flatten(), average='binary')

    # Calculate L1 & MSE
    l1, mse = None, None
    if pred_sdf is not None:
        l1 = torch.mean(torch.abs(gt_sdf - pred_sdf))
        mse = torch.mean((sdf2isdf(gt_sdf) - sdf2isdf(pred_sdf))**2)

    return dice, hausdorff, precision, recall, l1, mse, cldice, cldice_pcr, b0_errors, b1_errors


def eval_set(pred_path, seg_path, sdf_path=None):
    scores = {"VolDist": [],
              "DICE": [],
              "clDICE": [], 
              "Precision": [],
              "Recall": [],
              "clDICE(GC)": [],
              #"L1": [],
              #"MSE": [],
              "PT": [],
              "B0":[],
              "B1":[]}

    for pt in tqdm(os.listdir(f"{pred_path}/seg"), desc=f"Evaluating {pred_path.split('/')[-1]}"):
        if not pt.endswith('.nii.gz'):
            continue

        pred_seg, _ = torch_from_nii(f"{pred_path}/seg/{pt}")
        #pred_seg[pred_seg>=19] = 0
        #pred_seg[pred_seg==18] = 1
        pred_seg = pred_seg.to(torch.uint8)

        gt_seg_pcr, _ = torch_from_nii(f"/home/l.cai/small_bowel/small_bowel_data_code/motility_louis/isotropic/connectivity_2_scale05_cons15/cl_gc_excluded4/{pt}")#(f"/home/l.cai/small_bowel/small_bowel_data_code/motility_louis/isotropic/cl_gc_excluded4/{pt}")#(f"/home/l.cai/small_bowel/small_bowel_data_code/centerline_variation/cons10_scale5_connect18/{pt}")
        gt_seg_pcr = gt_seg_pcr.to(torch.uint8)
        pred_seg_pcr, _ = torch_from_nii(f"{pred_path}/cl_gc/{pt}")
        pred_seg_pcr = pred_seg_pcr.to(torch.uint8)
        print(pt)
        #print(np.unique(pred_seg.numpy()), type(pred_seg))
        if os.path.isdir(f"{pred_path}/sdf"):
            pred_sdf, _ = torch_from_nii(f"{pred_path}/sdf/{pt}")
        else:
            pred_sdf = None

        gt_seg, _ = torch_from_nii(f"{seg_path}/{pt}")
        gt_seg = gt_seg.to(torch.uint8)
        gt_sdf = None
        #print(np.unique(gt_seg.numpy()),type(gt_seg))
        if sdf_path is not None:
            gt_sdf, _ = torch_from_nii(f"{sdf_path}/{pt}")

        dice, hausdorff, precision, recall, l1, mse, cldice, cldice_pcr, b0,b1 = score(pred_seg, pred_sdf, pred_seg_pcr, gt_seg_pcr, gt_seg)
        

        # Update scores
        scores["DICE"].append(dice.item())
        scores["clDICE"].append(cldice)
        scores["clDICE(GC)"].append(cldice_pcr)
        scores["VolDist"].append(hausdorff.item())
        scores["Precision"].append(precision)
        scores["Recall"].append(recall)
        scores['PT'].append(pt)
        scores['B0'].append(b0)
        scores['B1'].append(b1)

        # if l1:
        #     scores["L1"].append(l1)
        # if mse:
        #     scores["MSE"].append(mse)

    print(pt, scores)
    del scores['PT']

    for k, v in scores.items():
        scores[k] = np.around(np.array(v, dtype=float).mean(), 4) if v else np.nan

    return scores


def eval_model(preds_path, seg_path, sdf_path=None):
    model_scores = dict()

    #model_scores['train'] = eval_set(f"{preds_path}/Train", seg_path, sdf_path)
    #model_scores['val'] = eval_set(f"{preds_path}/Val", seg_path, sdf_path)
    model_scores['test'] = eval_set(f"{preds_path}/Test", seg_path, sdf_path)

    return model_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True, help="Path to model's prediction folder")
    parser.add_argument("--gt_seg_path", default="../../Results/GTs/seg/", help="Path to segmentation ground truth folder")
    parser.add_argument("--gt_sdf_path", required=False, help="Path to sdf ground truth folder")

    args = parser.parse_args()

    results = eval_model(args.prediction_path, args.gt_seg_path, args.gt_sdf_path)
    print(results)

    with open(f"{args.prediction_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)
