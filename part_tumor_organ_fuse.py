from glob import glob
import SimpleITK as sitk
import numpy as np
import os
image_path = '/root/infersTr_tumor/*.nii.gz'

path = glob(image_path)
k = 0
for file in path:
    part_label_path = file.replace('infersTr_tumor', 'labelsTr')
    organ_label_path = file.replace('infersTr_tumor', 'imagesTr2200-pseudo-labels')
    save_path = file.replace('labelsTr', 'labels_2200')
    
    label_tumor = sitk.ReadImage(file)  # tumor pseudo-label
    label_part = sitk.ReadImage(part_label_path)  # ground truth partial-label
    label_organ = sitk.ReadImage(organ_label_path)  # organs pseudo-label
    
    tumor_array = sitk.GetArrayFromImage(label_tumor)
    part_label_array = sitk.GetArrayFromImage(label_part)
    organ_label_array = sitk.GetArrayFromImage(label_organ)
    
    
    # fusion part and tumor
    tumor_array[tumor_array != 0] = 14
    tumor_array[tumor_array == 0] = label_array[tumor_array == 0]    
    
    
    # fusion part-tumor and organ
    organ_label_array[organ_label_array == 1] = 0  # if this sample has 1, 2, 3, 4, 13 organs annotations
    organ_label_array[organ_label_array == 2] = 0
    organ_label_array[organ_label_array == 3] = 0
    organ_label_array[organ_label_array == 4] = 0
    organ_label_array[organ_label_array == 13] = 0
    
    tumor_array[tumor_array == 0] = organ_label_array[tumor_array == 0]
    
    
    new_label = sitk.GetImageFromArray(tumor_array)
    sitk.WriteImage(new_label, save_path)
    k = k + 1
    print(k)
    
        
    