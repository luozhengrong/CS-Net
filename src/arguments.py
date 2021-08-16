import argparse
GPU="0"
# [DATA]
DATASET_NAME = "CVLAB"
TRAIN_IMAGE_PATH = r"../cvlabData/trainCvlab/img/"
TRAIN_LABEL_PATH = r"../cvlabData/trainCvlab/lab/"

VAL_IMAGE_PATH = r"../cvlabData/testCvlab/img/"
VAL_LABEL_PATH = r"../cvlabData/testCvlab/lab/"

TEST_IMAGE_PATH = r"../cvlabData/testCvlab/img/"
TEST_LABEL_PATH = r"../cvlabData/testCvlab/lab/"

IN_SIZE = [512,512]
OUT_SIZE = [512,512]
PAD=0                     #  ********

BATCH_SIZE = 3
WORKER_NUM = 10
IMAGE_NUM=5                 #  ********
NOR=1                       #  ********
IN_CHANNELS = IMAGE_NUM*4
OUT_CHANNELS = 2

# [NETWORK]
USEALLGPU = True
GPU_DEVICE = [0]
CLASS_NUM = 2
BRANCH=7                    #  ********
SINGLE_BRANCH_CHANNEL= 10   #  ********
NUM_FILTERS=BRANCH*SINGLE_BRANCH_CHANNEL

# [TRAINING]
LEARNING_RATE = 0.0005
START_EPOCH = 0
END_EPOCH = 1750
SNAPSHOT_EPOCH = 10   # save model
VAL_EPOCH = 10      # val test

DECAY_STEP = 30
DECAY_RATE = 0.9

# FOR LOAD THE TRAINED MODEL
RMS="CS_Net_1"

EPOCH_MODEL_SAVE_PREFIX = "./history/RMS/saved_models3/model_epoch_"
ITERA_MODEL_SAVE_PREFIX = "./history/RMS/itera_saved_models3/model_itera_"


VAL_SEG_CSV_PATH = "../history/"+RMS+"/history_"+RMS+".csv"
TEST_SEG_CSV_PATH = "../history/"+RMS+"/history_test_"+RMS+".csv"
SAVE_DIR_PATH =  "../history/"+RMS


MODEL_SAVE_PATH = "../history/"+RMS+"/saved_models_"+RMS
IMAGE_SAVE_PATH = "../history/"+RMS+"/result_images_"+RMS

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Luo Network")
    parser.add_argument('-gpu', '--gpu', default=GPU, type=str,
                        help='Supprot one GPU & multiple GPUs.')
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME,
                        help="The name of the dataset.")
    parser.add_argument("--train-image-path", type=str, default=TRAIN_IMAGE_PATH,
                        help="Path to the directory containing the train image.")
    parser.add_argument("--train-label-path", type=str, default=TRAIN_LABEL_PATH,
                        help="Path to the directory containing the train label.")
    parser.add_argument("--val-image-path", type=str, default=VAL_IMAGE_PATH,
                        help="Path to the directory containing the validation image.")
    parser.add_argument("--val-label-path", type=str, default=VAL_LABEL_PATH,
                        help="Path to the directory containing the validation label.")

    parser.add_argument("--test-image-path", type=str, default=TEST_IMAGE_PATH,
                        help="Path to the directory containing the validation image.")
    parser.add_argument("--test-label-path", type=str, default=TEST_LABEL_PATH,
                        help="Path to the directory containing the validation label.")

    parser.add_argument("--in-size", type=int, default=IN_SIZE,
                        help="The input patch size of the volume.")
    parser.add_argument("--out-size", type=int, default=OUT_SIZE,
                        help="The input patch size of the volume.")
    parser.add_argument("--pad", type=int, default=PAD,
                        help="The input patch size of the volume.")
    parser.add_argument("--image-num", type=int, default=IMAGE_NUM,
                        help="The number of the input images.")
    parser.add_argument("--nor", type=int, default=NOR,
                        help="nor.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--in-channels", type=int, default=IN_CHANNELS,
                        help="")
    parser.add_argument("--out-channels", type=int, default=OUT_CHANNELS,
                        help="")
    parser.add_argument("--branch", type=int, default=BRANCH,
                        help="")
    parser.add_argument("--worker-num", type=int, default=WORKER_NUM,
                        help="")
    parser.add_argument("--num-filters", type=int, default=NUM_FILTERS,
                        help="")
    parser.add_argument("--useallgpu", type=str, default=USEALLGPU,
                        help=".")
    parser.add_argument("--gpu-device", type=str, default=GPU_DEVICE,
                        help=".")
    parser.add_argument("--class-num", type=int, default=CLASS_NUM,
                        help="Path to the file listing the images in the target dataset.")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="The start epoch.")
    parser.add_argument("--end-epoch", type=int, default=END_EPOCH,
                        help="The end epoch.")
    parser.add_argument("--snapshot-epoch", type=int, default=SNAPSHOT_EPOCH,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--val-epoch", type=int, default=SNAPSHOT_EPOCH,
                        help="Validation summaries and checkpoint every often..")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--decay-step", type=int, default=DECAY_STEP,
                        help="The step of regularisation parameter for L2-loss.")
    parser.add_argument("--epoch-model-save-prefix", type=str, default=EPOCH_MODEL_SAVE_PREFIX,
                        help="The prefix name of model save by epoch.")
    parser.add_argument("--itera-model-save-prefix", type=str, default=ITERA_MODEL_SAVE_PREFIX,
                        help="The prefix name of model save by iteration.")
    parser.add_argument("--val-seg-csv-path", type=str, default=VAL_SEG_CSV_PATH,
                        help="Where to save the validation csv file.")
    parser.add_argument("--save-dir-path", type=str, default=SAVE_DIR_PATH,
                        help="Where to save the file.")
    parser.add_argument("--test-seg-csv-path", type=str, default=TEST_SEG_CSV_PATH,
                        help="Where to save the HARD validation csv file.")
    parser.add_argument("--model-save-path", type=str, default=MODEL_SAVE_PATH,
                        help="Where to save the model.")
    parser.add_argument("--image-save-path", type=str, default=IMAGE_SAVE_PATH,
                        help="Where to save the image.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(args.train_image_path)
