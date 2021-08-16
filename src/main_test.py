# from HDC4 import CleanU_Net
from dataset import SEMDataTest
import torch
from metrics import *
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import imageio
from PIL import Image
# from modules import *
from save_history import *
from losses import *
import csv
import time
import re
from skimage import measure
from arguments import get_arguments

device = torch.device("cuda")


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


def save_prediction_image(stacked_img, im_name, epoch, save_folder_name="result_images", output_shape=[768, 1024],
                          save_im=True):
    img_cont_np = stacked_img.cpu().data.numpy().astype('uint8')
    img_cont_nps = measure.label(img_cont_np)
    number = img_cont_nps.max()
    for i in range(number):
        if np.count_nonzero(img_cont_nps == i) < 300:
            img_cont_np[img_cont_nps == i] = 0
    img_cont = img_cont_np * 255
    desired_path = save_folder_name + '/epoch_' + str(epoch) + "_test" + '/'
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = str(im_name) + '.png'
    imageio.imwrite(desired_path + export_name, img_cont[0])
    return img_cont_np

import nibabel as nib
def validate_model2(model, data_val, criterion, epoch, header2, save_dir, save_file_name2,
                    save_folder_name='prediction', pad=0, testflip=2):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    value = [None] * 167
    value[0] = epoch
    value[1] = str(testflip)
    num = 0
    total_val_acc = 0
    total_val_loss = 0
    total_val_dice = 0
    total_val_jac = 0
    total_val_Sp = 0
    total_val_sn = 0
    total_val_recall = 0
    total_dict = {}
    nii = np.zeros(shape=(165, 2, 768, 1024), dtype=np.float32)
    nii = torch.from_numpy(nii).to(device)
    for batch, (images_v, index) in enumerate(data_val):
        with torch.no_grad():
            image_v = images_v.to(device)
            index=int(index)
            # mask_v = masks_v.to(device)
            if testflip == 0:
                # print(i)
                logit = F.softmax(model(image_v, ori)[0], 1)
                logit += F.softmax(model(image_v.flip(dims=(2,)), ori.flip(dims=(2,)))[0].flip(dims=(2,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(3,)), ori.flip(dims=(3,)))[0].flip(dims=(3,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(2, 3)), ori.flip(dims=(2, 3)))[0].flip(dims=(2, 3)), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2), ori.permute(0, 1, 3, 2))[0].permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(2,)), ori.permute(0, 1, 3, 2).flip(dims=(2,)))[
                        0].flip(dims=(2,)).permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(3,)), ori.permute(0, 1, 3, 2).flip(dims=(3,)))[
                        0].flip(dims=(3,)).permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(2, 3)), ori.permute(0, 1, 3, 2).flip(dims=(2, 3)))[
                        0].flip(dims=(2, 3)).permute(0, 1, 3, 2), 1)
                output_v = logit / 8.0
                # logit = F.softmax(model(image_v)[i], 1)
                # logit += F.softmax(model(image_v.flip(dims=(2,)))[i].flip(dims=(2,)), 1)
                # logit += F.softmax(model(image_v.flip(dims=(3,)))[i].flip(dims=(3,)), 1)
                # logit += F.softmax(model(image_v.flip(dims=(2, 3)))[i].flip(dims=(2, 3)), 1)
                # logit += F.softmax(model(image_v.permute(0, 1, 3, 2))[i].permute(0, 1, 3, 2), 1)
                # logit += F.softmax(model(image_v.permute(0, 1, 3, 2).flip(dims=(2,)))[i].flip(dims=(2,)).permute(0, 1, 3, 2), 1)
                # logit += F.softmax(model(image_v.permute(0, 1, 3, 2).flip(dims=(3,)))[i].flip(dims=(3,)).permute(0, 1, 3, 2), 1)
                # logit += F.softmax(model(image_v.permute(0, 1, 3, 2).flip(dims=(2, 3)))[i].flip(dims=(2, 3)).permute(0, 1, 3, 2), 1)
                # output_v = logit / 8.0
            elif testflip == 1:
                # print(image_v.shape)
                logit = F.softmax(model(image_v)[1], 1)
                logit += F.softmax(model(image_v.flip(dims=(2,)))[1].flip(dims=(2,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(3,)))[1].flip(dims=(3,)), 1)
                logit += F.softmax(model(image_v.flip(dims=(2, 3)))[1].flip(dims=(2, 3)), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2))[1].permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(2,)))[
                        1].flip(dims=(2,)).permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(3,)))[
                        1].flip(dims=(3,)).permute(0, 1, 3, 2), 1)
                logit += F.softmax(
                    model(image_v.permute(0, 1, 3, 2).flip(dims=(2, 3)))[
                        1].flip(dims=(2, 3)).permute(0, 1, 3, 2), 1)
                output_v = logit / 8.0
            elif testflip == 2:
                out0_0, out1_0, out2_0,_,_,_ = model(image_v)
                out0_1, out1_1, out2_1,_,_,_ = model(image_v.flip(dims=(2,)))
                out0_1=out0_1.flip(dims=(2,))
                out1_1 = out1_1.flip(dims=(2,))
                out2_1 = out2_1.flip(dims=(2,))
                out0_2, out1_2, out2_2,_,_,_ = model(image_v.flip(dims=(3,)))
                out0_2=out0_2.flip(dims=(3,))
                out1_2 = out1_2.flip(dims=(3,))
                out2_2 = out2_2.flip(dims=(3,))
                out0_3, out1_3, out2_3,_,_,_ = model(image_v.flip(dims=(2, 3)))
                out0_3=out0_3.flip(dims=(2, 3))
                out1_3 = out1_3.flip(dims=(2, 3))
                out2_3 = out2_3.flip(dims=(2, 3))
                out0_4, out1_4, out2_4,_,_,_ = model(image_v.permute(0, 1, 3, 2))
                out0_4=out0_4.permute(0, 1, 3, 2)
                out1_4 = out1_4.permute(0, 1, 3, 2)
                out2_4 = out2_4.permute(0, 1, 3, 2)
                out0_5, out1_5, out2_5,_,_,_ = model(image_v.permute(0, 1, 3, 2).flip(dims=(2,)))
                out0_5=out0_5.flip(dims=(2,)).permute(0, 1, 3, 2)
                out1_5 = out1_5.flip(dims=(2,)).permute(0, 1, 3, 2)
                out2_5 = out2_5.flip(dims=(2,)).permute(0, 1, 3, 2)
                out0_6, out1_6, out2_6,_,_,_ = model(image_v.permute(0, 1, 3, 2).flip(dims=(3,)))
                out0_6=out0_6.flip(dims=(3,)).permute(0, 1, 3, 2)
                out1_6 = out1_6.flip(dims=(3,)).permute(0, 1, 3, 2)
                out2_6 = out2_6.flip(dims=(3,)).permute(0, 1, 3, 2)
                out0_7, out1_7, out2_7,_,_,_ = model(image_v.permute(0, 1, 3, 2).flip(dims=(2, 3)))
                out0_7=out0_7.flip(dims=(2, 3)).permute(0, 1, 3, 2)
                out1_7 = out1_7.flip(dims=(2, 3)).permute(0, 1, 3, 2)
                out2_7 = out2_7.flip(dims=(2, 3)).permute(0, 1, 3, 2)

                # import matplotlib.pyplot as plt
                # plt.imshow(out0_7[0,1,:,:].cpu().numpy())
                # plt.show()
                output_0=F.softmax(out0_0, 1)
                output_0 += F.softmax(out0_1, 1)
                output_0 += F.softmax(out0_2, 1)
                output_0 += F.softmax(out0_3, 1)
                output_0 += F.softmax(out0_4, 1)
                output_0 += F.softmax(out0_5, 1)
                output_0 += F.softmax(out0_6, 1)
                output_0 += F.softmax(out0_7, 1)

                output_1=F.softmax(out1_0, 1)
                output_1 += F.softmax(out1_1, 1)
                output_1 += F.softmax(out1_2, 1)
                output_1 += F.softmax(out1_3, 1)
                output_1 += F.softmax(out1_4, 1)
                output_1 += F.softmax(out1_5, 1)
                output_1 += F.softmax(out1_6, 1)
                output_1 += F.softmax(out1_7, 1)

                output_2 = F.softmax(out2_0, 1)
                output_2 += F.softmax(out2_1, 1)
                output_2 += F.softmax(out2_2, 1)
                output_2 += F.softmax(out2_3, 1)
                output_2 += F.softmax(out2_4, 1)
                output_2 += F.softmax(out2_5, 1)
                output_2 += F.softmax(out2_6, 1)
                output_2 += F.softmax(out2_7, 1)

                logit0 = output_0 / 8.0
                logit1 = output_1 / 8.0
                logit2 = output_2 / 8.0
                # print(image_v.shape)
                # logit = F.softmax(model(image_v), 1)
                # logit += F.softmax(model(image_v.flip(dims=(2,))).flip(dims=(2,)), 1)
                # logit += F.softmax(model(image_v.flip(dims=(3,))).flip(dims=(3,)), 1)
                # logit += F.softmax(model(image_v.flip(dims=(2, 3))).flip(dims=(2, 3)), 1)
                # logit += F.softmax(model(image_v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2), 1)
                # logit += F.softmax(
                #     model(image_v.permute(0, 1, 3, 2).flip(dims=(2,))).flip(dims=(2,)).permute(0, 1, 3, 2), 1)
                # logit += F.softmax(
                #     model(image_v.permute(0, 1, 3, 2).flip(dims=(3,))).flip(dims=(3,)).permute(0, 1, 3, 2), 1)
                # logit += F.softmax(
                #     model(image_v.permute(0, 1, 3, 2).flip(dims=(2, 3))).flip(dims=(2, 3)).permute(0, 1, 3, 2), 1)
                # output_v = logit / 8.0
            else:
                out0, out1, out2,_,_,_ = model(image_v)
                logit0 = F.softmax(out0, 1)
                logit1 = F.softmax(out1, 1)
                logit2 = F.softmax(out2, 1)
            if pad != 0:
                output_v = output_v[:, :, pad:-pad, pad:-pad]
        # nii[index] = nii[index] + logit1
        if index == 0:
            nii[index] = nii[index] + logit1
            nii[index + 1] = nii[index + 1] + logit2
        elif index == 164:
            nii[index - 1] = nii[index - 1] + logit0
            nii[index] = nii[index] + logit1
        else:
            nii[index - 1] = nii[index - 1] + logit0
            nii[index] = nii[index] + logit1
            nii[index + 1] = nii[index + 1] + logit2
        print(batch)
    nii=torch.argmax(nii, dim=1).cpu().numpy().astype('uint8')
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    nib.save(nib.Nifti1Image(nii, None), desired_path + str(epoch) + "_ori.nii.gz")
    labels=measure.label(nii)
    for i in range(1,labels.max()+1):
        if np.count_nonzero(labels==i)<3000:
            print(np.count_nonzero(labels==i))
            nii[labels==i]=0

    nib.save(nib.Nifti1Image(nii, None), desired_path+str(epoch)+".nii.gz")
    lab=np.array(nib.load("lab.nii.gz").get_fdata())/255
    for j in range(165):
        jac = dice_coeff(nii[j,:,:], lab[j,:,:])
        value[j + 2] = (jac["jac"])
        total_val_dice = total_val_dice + jac["f1"]
        total_val_jac = total_val_jac + jac["jac"]
        total_val_Sp = total_val_Sp + jac["sp"]
        total_val_sn = total_val_sn + jac["sn"]
        total_val_recall = total_val_recall + jac["rc"]
        total_val_acc = total_val_acc + jac["acc"]
        num +=1
    export_history(header2, value, save_dir, save_file_name2)
    total_dict['dice'] = total_val_dice
    total_dict['jac'] = total_val_jac
    total_dict['sp'] = total_val_Sp
    total_dict['sn'] = total_val_sn
    total_dict['recall'] = total_val_recall
    total_dict['acc'] = total_val_acc
    from post_luo import post_process
    post_process(nii, lab)

    all_jac=dice_coeff(nii, lab)

    return total_dict, num,all_jac["jac"],all_jac["f1"]


if __name__ == "__main__":
    args = get_arguments()
    flip = 2

    pad_size = args.pad
    epoch = 0
    model_dir = args.model_save_path + "/model_epoch_" + str(epoch) + ".pwf"
    model = torch.load(model_dir)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
    criterion = jacc_loss()
    SEM_val = SEMDataTest(image_path=args.test_image_path,
                          mask_path=args.test_label_path,
                          pad=pad_size,
                          image_num=args.image_num,
                          nor=args.nor)
    # Dataset end
    header2 = ["epoch", "flip"]
    for i in range(165):
        header2.append(str(i + 1))

    save_file_name2 = args.test_seg_csv_path
    save_dir = args.save_dir_path
    image_save_path = args.image_save_path
    # Dataloader begins
    if flip == 0:
        num_flip = 3
    elif flip == 1:
        num_flip = 1
    elif flip == 2:
        num_flip = 2
    else:
        num_flip = 4
        print("num_flip mistake")
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=6, batch_size=1)
    total_dict, num,ajac,adice = validate_model2(model, SEM_val_load, criterion, epoch + num_flip, header2, save_dir,
                                      save_file_name2,
                                      image_save_path, pad_size, testflip=flip)

    print('F1:', total_dict["dice"] / (num), "JAC:", total_dict["jac"] / (num))
    print("ajac:",ajac,"adice:",adice)
    print('Sp:', total_dict["sp"] / (num), "Sn:", total_dict["sn"] / (num))
    print('Recall:', total_dict["recall"] / (num), "ACC:", total_dict["acc"] / (num))
