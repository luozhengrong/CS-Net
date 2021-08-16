from CS_Net import cs_net
from dataset import SEMDataTrain, SEMDataVal
from torch.autograd import Variable
import numpy as np
from save_history import *
from losses import *
import time
from val import validate_model2
from arguments import get_arguments
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def main():
    args = get_arguments()
    # Dataset begin
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    SEM_train = SEMDataTrain(image_path=args.train_image_path,
                             mask_path=args.train_label_path,
                             in_size=args.in_size,
                             out_size=args.out_size,
                             pad=args.pad,
                             image_num=args.image_num,
                             nor=args.nor)
    SEM_val = SEMDataVal(image_path=args.val_image_path,
                         mask_path=args.val_label_path,
                         in_size=args.in_size,
                         out_size=args.out_size,
                         pad=args.pad,
                         image_num=args.image_num,
                         nor=args.nor)
    # Dataset end

    # Dataloader begins
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=args.worker_num, batch_size=args.batch_size, shuffle=True)
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=args.worker_num, batch_size=1, shuffle=False)
    # Dataloader end

    # Model
    model = cs_net(in_channels=args.in_channels, out_channels=args.out_channels, num_filters=args.num_filters)
    if args.useallgpu:
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpuDevice)
    # Loss function

    criterion = Instance_Aware_Loss()
    criterion4 = nn.BCEWithLogitsLoss()
    # Optimizerd
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=2, eta_min=0.000001)
    # Parameters
    epoch_start = args.start_epoch
    epoch_end = args.end_epoch
    header2 = ["epoch"]
    for i in range(165):
        header2.append(str(i + 1))

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val dice', 'val jac']
    save_file_name = args.val_seg_csv_path
    save_file_name2 = args.test_seg_csv_path
    save_dir = args.save_dir_path

    # Saving images and models directories
    model_save_dir = args.model_save_path
    image_save_path = args.image_save_path
    max_jac = 0
    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # adjust_learning_rate(optimizer,i,epoch_end,lr)
        # train the model
        start_time = time.time()
        model.train()
        for batch, (images, masks0, weight0, dist1, masks1, weight1, masks2, weight2,) in enumerate(SEM_train_load):
            images = Variable(images.cuda())
            masks0 = Variable(masks0.cuda())
            masks1 = Variable(masks1.cuda())
            masks2 = Variable(masks2.cuda())

            weight0 = Variable(weight0.cuda())
            weight1 = Variable(weight1.cuda())
            weight2 = Variable(weight2.cuda())

            # dist0 = Variable(dist0.cuda())
            dist1 = Variable(dist1.cuda())
            # dist2 = Variable(dist2.cuda())

            outputs = model(images)

            loss0 = criterion(outputs[0], masks0, weight0)
            loss1 = criterion(outputs[1], masks1, weight1)
            loss2 = criterion(outputs[2], masks2, weight2)

            # loss0_2 = criterion4(outputs[3],dist0.unsqueeze(1))
            loss1_2 = criterion4(outputs[4], dist1.unsqueeze(1))
            # loss2_2 = criterion4(outputs[5], dist2.unsqueeze(1))
            loss = (loss0 + loss1 + loss2) + (loss1_2)
            L_1 = loss0.cpu().item()
            L_2 = loss1.cpu().item()
            L_3 = loss2.cpu().item()
            re_loss = loss1_2.cpu().item()
            all_loss = loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # *************************************  #
        train_acc = 0
        # *************************************  #
        train_loss = all_loss
        over_time = time.time()

        scheduler.step()
        lr = scheduler.get_lr()

        print('Epoch', str(i + 1), "1_loss:", L_1, "2_loss:", L_2, "3_loss", L_3, "re loss:", re_loss, 'Train loss:',
              train_loss, "Train acc",
              train_acc, "time", (over_time - start_time), "lr", lr)

        # Validation every 5 epoch
        if (i + 1) >= 0:
            if (i + 1) % args.val_epoch == 0:
                start_time2 = time.time()
                total_dict, num, alljac, alldice = validate_model2(model, SEM_val_load, criterion, i + 1,
                                                                   image_save_path, testflip=False, pad=args.pad)
                over_time2 = time.time()
                print("time", (over_time2 - start_time2))
                print('F1:', total_dict["dice"] / (num), "JAC:", total_dict["jac"] / (num))
                print('Sp:', total_dict["sp"] / (num), "Sn:", total_dict["sn"] / (num))
                print('Recall:', total_dict["recall"] / (num), "ACC:", total_dict["acc"] / (num))
                values = [i + 1, train_loss, train_acc, total_dict["dice"] / (num), total_dict["jac"] / (num)]
                export_history(header, values, save_dir, save_file_name)
                if total_dict["jac"] / (num) > max_jac:
                    max_jac = total_dict["jac"] / (num)
                    save_models(model, model_save_dir, 0)
                print("max_jac:", max_jac)
                if (i + 1) % args.snapshot_epoch == 0:
                    save_models(model, model_save_dir, i + 1)


if __name__ == "__main__":
    main()
