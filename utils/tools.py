import os
import cv2
import numpy as np
from sklearn import metrics



def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1)):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)
    
    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    
    return FP, FN, TP, TN

def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1), mask_arr=None):
    # pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)
    
    return acc

def get_best_thresh(gt_arr, pred_arr):
    gt_arr = gt_arr
    pred_arr = pred_arr
    fpr, tpr, thresholds = metrics.roc_curve(gt_arr.reshape(-1), pred_arr.reshape(-1), pos_label=1)
    
    best_acc = 0
    thresh_value = 0
    for i in range(thresholds.shape[0]):
        thresh_arr = pred_arr.copy()
        thresh_arr[thresh_arr >= thresholds[i]] = 1
        thresh_arr[thresh_arr < thresholds[i]] = 0
        current_acc = calc_acc(thresh_arr, gt_arr)
        if current_acc >= best_acc:
            best_acc = current_acc
            thresh_value = thresholds[i]
    
    thresh_arr = pred_arr.copy()
    thresh_arr[thresh_arr >= thresh_value] = 255
    thresh_arr[thresh_arr < thresh_value] = 0
    
    return thresh_value
