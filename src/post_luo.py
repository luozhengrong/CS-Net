import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
def to_numpy(item):
    '''
    convert input to numpy array
    input type: image-file-name, PIL image, torch tensor, numpy array
    '''
    if 'str' in str(type(item)):  # read the image as numpy array
        item = np.array(Image.open(item))
    elif 'PIL' in str(type(item)):
        item = np.array(item)
    elif 'torch' in str(type(item)):
        item = item.numpy()
    elif 'numpy' in str(type(item)):
        pass
    else:  # unsupported type
        print('WTF:', str(type(item)))
        return None
    return item
def dice_coeff(pred, lable):
    # convert to numpy array
    pred = to_numpy(pred)
    lable = to_numpy(lable)

    # convert to 1-D array for convinience
    pred = pred.flatten()
    lable = lable.flatten()
    # convert to 0-1 array
    pred = np.uint8(pred != 0)
    lable = np.uint8(lable != 0)

    met_dict = {}  # metrics dictionary

    TP = np.count_nonzero((pred + lable) == 2)  # true positive
    TN = np.count_nonzero((pred + lable) == 0)  # true negative
    FP = np.count_nonzero(pred > lable)  # false positive
    FN = np.count_nonzero(pred < lable)  # false negative

    smooth = 1e-9  # avoid devide zero
    acc = (TP + TN) / (TP + TN + FP + FN + smooth)  # accuracy
    sn = TP / (TP + FP + smooth)  # sensitivity, or precision
    sp = TN / (TN + FN + smooth)  # specificity
    rc = TP / (TP + FN + smooth)  # recall
    f1 = 2 * sn * rc / (sn + rc + smooth)  # F1 mesure
    jac = TP / (TP + FN + FP + smooth)  # jaccard coefficient

    # return metrics as dictionary
    met_dict['TP'] = TP
    met_dict['TN'] = TN
    met_dict['FP'] = FP
    met_dict['FN'] = FN
    met_dict['acc'] = acc
    met_dict['sn'] = sn
    met_dict['sp'] = sp
    met_dict['rc'] = rc
    met_dict['f1'] = f1
    met_dict['jac'] = jac
    return met_dict
def close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    test2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  #
    return test2
def fill_holes(img):
    test2 = ndimage.binary_fill_holes(img).astype(('uint8'))
    return test2
def medianBlur(imgs):
    seg_img1 = []
    x,y,z=imgs.shape
    test = imgs.transpose(2, 1, 0)
    for j in range(z):
        test2 = cv2.medianBlur(test[j, :, :], 5)
        seg_img1.append(test2)
    seg_img1 = np.array(seg_img1)
    seg_img1 = seg_img1.transpose(2, 1, 0)
    return seg_img1


def medianBlur2(imgs):
    seg_img1 = []
    x,y,z=imgs.shape
    test = imgs.transpose(1, 0, 2)
    for j in range(y):
        test2 = cv2.medianBlur(test[j, :, :], 5)
        seg_img1.append(test2)
    seg_img1 = np.array(seg_img1)
    seg_img1 = seg_img1.transpose(1, 0, 2)
    return seg_img1

def metric(seg_img,lab):
    jac_all = 0
    dice_all = 0
    for k in range(165):
        dict = dice_coeff(seg_img[k, :, :], lab[k, :, :])
        jac_all = jac_all + dict["jac"]
        dice_all = dice_all + dict['f1']
    single_jac = (jac_all / 165)
    single_dice = (dice_all / 165)
    print("single jac:", single_jac)
    print("single dice:", single_dice)
    dict = dice_coeff(seg_img, lab)
    all_jac = (dict["jac"])
    all_dice = dict['f1']
    print("all jac:", (all_jac))
    print("all dice:", (all_dice))
def post_process(seg="",lab="",save_path='com.nii.gz',close_img=False,fill=True,medi=True):
    # seg = np.array(nib.load(seg).get_data())
    # lab = np.array(nib.load(lab).get_data())/255
    assert seg.max()==1 and lab.max()==1,"not 0-1"
    z,y,x=lab.shape
    seg_img = []


    for i in range(z):
        img=seg[i,:,:]

        if close_img:
            img=close(img)
        if fill:
            img=np.pad(img,((0,64),(0,64)),"constant",constant_values=1.0)
            img=fill_holes(img)
            img=img[:-64,:-64]
        # kernel=np.ones((3,3),np.uint8)
        #img=cv2.erode(img,kernel)
        # img = cv2.dilate(img, kernel)
            # img=img[:,3:-3]
            # img = np.pad(img, ((0, 0), (3, 3)), "constant", constant_values=0)
        seg_img.append(img)
    seg = np.array(seg_img)
    if medi:
        #seg=medianBlur2(seg)
        seg=medianBlur(seg)
    print(np.unique(seg))
    print(np.unique(lab))
    metric(seg,lab)
    # print(seg.max(),lab.max())
    nib.save(nib.Nifti1Image(seg, None), save_path.replace(".nii","_ori.nii"))
    seg = lab + seg * 2
    seg[seg == 3] = 4
    seg[seg == 2] = 3
    seg[seg == 4] = 2
    nib.save(nib.Nifti1Image(seg, None), save_path)
    return seg

if __name__ == "__main__":
    pass
