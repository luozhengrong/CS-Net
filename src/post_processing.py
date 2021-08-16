import numpy as np
from matplotlib import pyplot as plt


def postprocess(image_path):
    ''' postprocessing of the prediction output
    Args
        image_path : path of the image
    Returns
        watershed_grayscale : numpy array of postprocessed image (in grayscale)
    '''

    # Bring in the image
    img_original = cv2.imread(image_path)
    img = cv2.imread(image_path)

    # In case the input image has 3 channels (RGB), convert to 1 channel (grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use threshold => Image will have values either 0 or 255 (black or white)
    ret, bin_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove Hole or noise through the use of opening, closing in Morphology module
    kernel = np.ones((1, 1), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)

    # remove noise in
    closing = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # make clear distinction of the background
    # Incerease/emphasize the white region.
    sure_bg = cv2.dilate(closing, kernel1, iterations=1)

    # calculate the distance to the closest zero pixel for each pixel of the source.
    # Adjust the threshold value with respect to the maximum distance. Lower threshold, more information.
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown is the region of background with foreground excluded.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # labelling on the foreground.
    ret, markers = cv2.connectedComponents(sure_fg)
    markers_plus1 = markers + 1
    markers_plus1[unknown == 255] = 0

    # Appy watershed and label the borders
    markers_watershed = cv2.watershed(img, markers_plus1)

    # See the watershed result in a clear white page.
    img_x, img_y = img_original.shape[0], img_original.shape[1]  # 512x512
    white, white_color = np.zeros((img_x, img_y, 3)), np.zeros((img_x, img_y, 3))
    white += 255
    white_color += 255
    # 1 in markers_watershed indicate the background value
    # label everything not indicated as background value
    white[markers_watershed != 1] = [0, 0, 0]  # grayscale version
    white_color[markers_watershed != 1] = [255, 0, 0]  # RGB version

    # Convert to numpy array for later processing
    white_np = np.asarray(white)  # 512x512x3
    watershed_grayscale = white_np.transpose(2, 0, 1)[0, :, :]  # convert to 1 channel (grayscale)
    img[markers_watershed != 1] = [255, 0, 0]

    return watershed_grayscale

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
    print("jac:", single_jac)
    print("dice:", single_dice)

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


if __name__ == '__main__':
    from PIL import Image

    print(postprocess('../data/train/masks/25.png'))
