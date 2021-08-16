import nibabel
import nibabel as nib
import numpy as np
from PIL import Image
import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件
ori=r"/media/mip/D/CS_Net/CS_Net/cvlabData/testCvlab/lab/"
seg_img=[]
for i in range(165):
    img_as_img = Image.open(ori+"test"+str(i+1).zfill(3)+".png")
    img_as_np = np.array(img_as_img)
    seg_img.append(img_as_np)
seg_img=np.array(seg_img)
nib.save(nib.Nifti1Image(seg_img, None), r"lab.nii.gz")