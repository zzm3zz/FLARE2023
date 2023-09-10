from glob import glob
import SimpleITK as sitk
import numpy as np
import os
image_path = '/root/infersTr_tumor/*.nii.gz'

path = glob(image_path)
k = 0
for file in path:
    organ_label_path = file.replace('infersTr_tumor', 'imagesTr2200-pseudo-labels')
    save_path = file.replace('labelsTr', 'labels_2200')
    
    label_tumor = sitk.ReadImage(file)  # tumor pseudo-label
    label_organ = sitk.ReadImage(organ_label_path)  # organs pseudo-label
    
    tumor_array = sitk.GetArrayFromImage(label_tumor)
    organ_label_array = sitk.GetArrayFromImage(label_organ)
    
    # fusion tumor and organ
    tumor_array[tumor_array == 0] = organ_label_array[tumor_array == 0]
    
    
    new_label = sitk.GetImageFromArray(tumor_array)
    sitk.WriteImage(new_label, new_path)
    k = k + 1
    print(k)
    
        
    