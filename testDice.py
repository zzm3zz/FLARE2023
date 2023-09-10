import argparse
import os
import shutil

import SimpleITK as sitk
import imageio
import numpy as np
from medpy import metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        # hd95=1
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    inferTsPath = '/root/infersTs/'
    labelTsPath = '/root/GT_50/'
    input_files = os.listdir(labelTsPath)
    Average_Dice_Liver = 0
    Average_Dice_Kidney_R = 0
    Average_Dice_Spleen = 0
    Average_Dice_Pancreas = 0
    Average_Dice_Aorta = 0
    Average_Dice_IVC = 0
    Average_Dice_RAG = 0
    Average_Dice_LAG = 0
    Average_Dice_Gallbladder = 0
    Average_Dice_Eso = 0
    Average_Dice_Stomach = 0
    Average_Dice_Duo = 0
    Average_Dice_Kidney_L = 0
    Average_Dice_Tumor = 0


    total_nii = 0
    for input_file in input_files:
        if input_file[-3:] == '.gz':
            total_nii += 1
            label_pred_path = inferTsPath + input_file
            label_gt_path = label_pred_path.replace('infersTs', 'GT_50')
            # label_gt_path = label_gt_path.replace('img', 'label')

            label_pred = sitk.ReadImage(label_pred_path)
            label_pred_array = sitk.GetArrayFromImage(label_pred)

            label_gt = sitk.ReadImage(label_gt_path)
            label_gt_array = sitk.GetArrayFromImage(label_gt)
            print(label_gt_array.shape)
            print(label_pred_array.shape)
            Liver_dice_hd = calculate_metric_percase(label_pred_array == 1, label_gt_array == 1)
            Kidney_R_dice_hd = calculate_metric_percase(label_pred_array == 2, label_gt_array == 2)
            Spleen_dice_hd = calculate_metric_percase(label_pred_array == 3, label_gt_array == 3)
            Pancreas_dice_hd = calculate_metric_percase(label_pred_array == 4, label_gt_array == 4)
            Aorta_dice_hd = calculate_metric_percase(label_pred_array == 5, label_gt_array == 5)
            IVC_dice_hd = calculate_metric_percase(label_pred_array == 6, label_gt_array == 6)
            RAG_dice_hd = calculate_metric_percase(label_pred_array == 7, label_gt_array == 7)
            LAG_dice_hd = calculate_metric_percase(label_pred_array == 8, label_gt_array == 8)            
            Gallbladder_dice_hd = calculate_metric_percase(label_pred_array == 9, label_gt_array == 9)
            Eso_dice_hd = calculate_metric_percase(label_pred_array == 10, label_gt_array == 10)
            Stomach_dice_hd = calculate_metric_percase(label_pred_array == 11, label_gt_array == 11)
            Duo_dice_hd = calculate_metric_percase(label_pred_array == 12, label_gt_array == 12)
            Kidney_L_dice_hd = calculate_metric_percase(label_pred_array == 13, label_gt_array == 13)
            Tumor_dice_hd = calculate_metric_percase(label_pred_array == 14, label_gt_array == 14)
            
            
            
            
            
            print('===============', total_nii)
            print('Dice of Liver is : ', Liver_dice_hd)
            print('Dice of Kidney_R is : ', Kidney_R_dice_hd)
            print('Dice of Spleen is : ', Spleen_dice_hd)
            print('Dice of Pancreas is : ', Pancreas_dice_hd)
            print('Dice of Aorta is : ', Aorta_dice_hd)
            print('Dice of IVC is : ', IVC_dice_hd)
            print('Dice of RAG is : ', RAG_dice_hd)
            print('Dice of LAG is : ', LAG_dice_hd)
            print('Dice of Gallbladder is : ', Gallbladder_dice_hd)
            print('Dice of Eso is : ', Eso_dice_hd)
            print('Dice of Stomach is : ', Stomach_dice_hd)
            print('Dice of Duo is : ', Duo_dice_hd)
            print('Dice of Kidney_L is : ', Kidney_L_dice_hd)
            print('Dice of Tumor is : ', Tumor_dice_hd)

            Average_Dice_Liver += Liver_dice_hd
            Average_Dice_Kidney_R += Kidney_R_dice_hd
            Average_Dice_Spleen += Spleen_dice_hd
            Average_Dice_Pancreas += Pancreas_dice_hd
            Average_Dice_Aorta += Aorta_dice_hd
            Average_Dice_IVC += IVC_dice_hd
            Average_Dice_RAG += RAG_dice_hd
            Average_Dice_LAG += LAG_dice_hd
            Average_Dice_Gallbladder += Gallbladder_dice_hd
            Average_Dice_Eso += Eso_dice_hd
            Average_Dice_Stomach += Stomach_dice_hd
            Average_Dice_Duo += Duo_dice_hd
            Average_Dice_Kidney_L += Kidney_L_dice_hd
            Average_Dice_Tumor += Tumor_dice_hd


            # Average_HD_Aorta += Aorta_dice_hd[1]
            # Average_HD_Gallbladder += Gallbladder_dice_hd[1]
            # Average_HD_Kidney_L += Kidney_L_dice_hd[1]
            # Average_HD_Kidney_R += Kidney_R_dice_hd[1]
            # Average_HD_Liver += Liver_dice_hd[1]
            # Average_HD_Pancreas += Pancreas_dice_hd[1]
            # Average_HD_Spleen += Spleen_dice_hd[1]
            # Average_HD_Stomach += Stomach_dice_hd[1]

    Average_Dice_Liver /= total_nii
    Average_Dice_Kidney_R /= total_nii
    Average_Dice_Spleen /= total_nii
    Average_Dice_Pancreas /= total_nii
    Average_Dice_Aorta /= total_nii
    Average_Dice_IVC /= total_nii
    Average_Dice_RAG /= total_nii
    Average_Dice_LAG /= total_nii
    Average_Dice_Gallbladder /= total_nii
    Average_Dice_Eso /= total_nii
    Average_Dice_Stomach /= total_nii
    Average_Dice_Duo /= total_nii
    Average_Dice_Kidney_L /= total_nii
    Average_Dice_Tumor /= total_nii

    # Average_HD_Aorta /= total_nii
    # Average_HD_Gallbladder /= total_nii
    # Average_HD_Kidney_L /= total_nii
    # Average_HD_Kidney_R /= total_nii
    # Average_HD_Liver /= total_nii
    # Average_HD_Pancreas /= total_nii
    # Average_HD_Spleen /= total_nii
    # Average_HD_Stomach /= total_nii

    print('===============')
    print('Average_Dice_Liver : ', Average_Dice_Liver)
    print('Average_Dice_Kidney_R : ', Average_Dice_Kidney_R)
    print('Average_Dice_Spleen : ', Average_Dice_Spleen)
    print('Average_Dice_Pancreas : ', Average_Dice_Pancreas)
    print('Average_Dice_Aorta : ', Average_Dice_Aorta)
    print('Average_Dice_IVC : ', Average_Dice_IVC)
    print('Average_Dice_RAG : ', Average_Dice_RAG)
    print('Average_Dice_LAG : ', Average_Dice_LAG)
    print('Average_Dice_Gallbladder : ', Average_Dice_Gallbladder)
    print('Average_Dice_Eso : ', Average_Dice_Eso)
    print('Average_Dice_Stomach : ', Average_Dice_Stomach)
    print('Average_Dice_Duo : ', Average_Dice_Duo)
    print('Average_Dice_Kidney_L : ', Average_Dice_Kidney_L)
    print('Average_Dice_Tumor : ', Average_Dice_Tumor)

    
    print('All average dice: ', (Average_Dice_Liver + Average_Dice_Kidney_R + Average_Dice_Spleen + Average_Dice_Pancreas + Average_Dice_Aorta + Average_Dice_IVC + Average_Dice_RAG + Average_Dice_LAG + Average_Dice_Gallbladder + Average_Dice_Eso + Average_Dice_Stomach + Average_Dice_Duo + Average_Dice_Kidney_L) / 13)
    print('All average dice: ', (Average_Dice_Liver + Average_Dice_Kidney_R + Average_Dice_Spleen + Average_Dice_Pancreas + Average_Dice_Aorta + Average_Dice_IVC + Average_Dice_RAG + Average_Dice_LAG + Average_Dice_Gallbladder + Average_Dice_Eso + Average_Dice_Stomach + Average_Dice_Duo + Average_Dice_Kidney_L + Average_Dice_Tumor) / 14)
   
    # print('All average hd95: ', (Average_HD_Aorta + Average_HD_Gallbladder + Average_HD_Kidney_L + Average_HD_Kidney_R + Average_HD_Liver + Average_HD_Pancreas + Average_HD_Spleen + Average_HD_Stomach) / 8)
    # image_name = str(current_slice) + 'gsrgsr.png'
    # for j in range(0, label_pred_array.shape[1]):
    #     for k in range(0, label_pred_array.shape[2]):
    #         if label_pred_slice[j][k] == 3:
    #             label_pred_slice[j][k] = 200
    # imageio.imwrite(image_name, label_pred_slice)
    # shutil.move(image_name, labelTsPath)
