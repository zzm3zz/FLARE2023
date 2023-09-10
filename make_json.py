import os
from glob import glob


image_path = '/root/imagesTr/*.nii.gz'
txt_path = '/root/data.txt'
path = glob(image_path)
for file in path:
    json_path = ''
    pre_ = '{"image":".'
    mid_ = '","label":".'
    post_ = '"},'
    image_name = file[-34:]
    image_name = image_name.replace('_0000.nii.gz', '.nii.gz')
    label_name = image_name.replace('imagesTr', 'labelsTr')
    json_path = pre_ + image_name + mid_ + label_name + post_ + '\n'
    f = open(txt_path, 'a')
    f.write(json_path)
    f.close()
    
