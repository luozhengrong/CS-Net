"""Common image segmentation metrics.
"""

import torch
import numpy as np
from PIL import Image
#from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

EPS = 1e-10


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _fast_hist(true, pred, num_classes):
    # pred = pred.float()
    # true = true.float()
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


# def jaccard_index(hist):
#     """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
#
#     Args:
#         hist: confusion matrix.
#
#     Returns:
#         avg_jacc: the average per-class jaccard index.
#     """
#     A_inter_B = torch.diag(hist)
#     A = hist.sum(dim=1)
#     B = hist.sum(dim=0)
#     jaccard = A_inter_B / (A + B - A_inter_B + EPS)
#     avg_jacc = nanmean(jaccard)
#     return avg_jacc
#
#
# def dice_coefficient(hist):
#     """Computes the SÃ¸rensenâ€“Dice coefficient, a.k.a the F1 score.
#
#     Args:
#         hist: confusion matrix.
#
#     Returns:
#         avg_dice: the average per-class dice coefficient.
#     """
#     A_inter_B = torch.diag(hist)
#     A = hist.sum(dim=1)
#     B = hist.sum(dim=0)
#     dice = (2 * A_inter_B) / (A + B + EPS)
#     avg_dice = nanmean(dice)
#     return avg_dice
#
#
# def eval_metrics(true, pred, num_classes):
#     """Computes various segmentation metrics on 2D feature maps.
#
#     Args:
#         true: a tensor of shape [B, H, W] or [B, 1, H, W].
#         pred: a tensor of shape [B, H, W] or [B, 1, H, W].
#         num_classes: the number of classes to segment. This number
#             should be less than the ID of the ignored class.
#
#     Returns:
#         overall_acc: the overall pixel accuracy.
#         avg_per_class_acc: the average per-class pixel accuracy.
#         avg_jacc: the jaccard index.
#         avg_dice: the dice coefficient.
#     """
#     # true = true.float()
#     # pred = pred.float()
#     # true = torch.from_numpy(true)
#     pred = pred[np.newaxis, :]
#
#     true = true.int()
#     pred = pred.int()
#     hist = torch.zeros((num_classes, num_classes))
#     for t, p in zip(true, pred):
#         hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
#     overall_acc = overall_pixel_accuracy(hist)
#     avg_per_class_acc = per_class_pixel_accuracy(hist)
#     avg_jacc = jaccard_index(hist)
#     avg_dice = dice_coefficient(hist)
#     return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coeff2(pred, target):
    ims = [pred, target]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    pred = np_ims[0]
    target = np_ims[1]

    smooth = 0.000001

    m1 = pred.flatten()  # Flatten
    m2 = target.flatten()  # Flatten
    tt = m1 * m2
    intersection = (m1 * m2).sum()
    intersection = np.float(intersection)

    bing = (np.uint8(m1) | np.uint8(m2)).sum()
    bing = bing.astype('float')
    jac = intersection / bing

    ff1 = 2 * jac / (1 + jac)

    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    return dice, jac


def dice_coeff_checkforBatch(pred, target, batch_size):
    batch_dice = 0
    batch_jac = 0
    for index in range(batch_size):
        dice, jac = dice_coeff(pred[index, ...], target[index, ...])
        batch_dice += dice
        batch_jac += jac
    return batch_dice / batch_size, batch_jac / batch_size


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


def roc(lable, pred):
    # convert to numpy array
    lable = to_numpy(lable)
    pred = to_numpy(pred)
    # convert to 1-D array for convinience
    pred = pred.flatten()
    lable = lable.flatten()
    # convert lable to 0-1 array
    lable = np.uint8(lable != 0)
    fpr, tpr, thresholds = roc_curve(lable, pred)
    auc_roc = auc(fpr, tpr)  # area under curve
    # plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc.jpg')
    plt.show()
    return auc_roc


if __name__ == "__main__":
    # t1 = torch.rand((5, 4, 2))
    # t1 = t1 > 0.5
    # t2 = torch.rand((5, 4, 2))
    # t2 = t2 > 0.5
    # t1 = t1.int()
    # t2 = t2.int()
    # overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(t2, t1, 2)
    # print("acc:{0} , perclassacc:{1}, jcc:{2}, dice:{3}" .format(overall_acc, avg_per_class_acc, avg_jacc, avg_dice))
    t1 = torch.rand((5, 4, 2))
    t1 = t1 > 0.5
    t2 = torch.rand((5, 4, 2))
    t2 = t2 > 0.5
    t1 = t1.int()
    t2 = t2.int()
    dice = dice_coeff(t1, t2)
    print(dice)