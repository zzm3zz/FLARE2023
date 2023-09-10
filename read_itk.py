import nibabel as nib
from glob import glob
# import cv2
from PIL import Image
import numpy as np
import SimpleITK as sitk

image_path = '/root/*.nii.gz'
path = glob(image_path)
n = 0
for file in path:
        try:
            img = sitk.ReadImage(file)
            print('true')
        
        except:
            # print(file)
            print('this is not true orient')
            img = nib.load(file) 
            qform = img.get_qform()
            img.set_qform(qform)
            sfrom = img.get_sform()
            img.set_sform(sfrom)
            nib.save(img, file)
            
        n = n + 1
        print(n)
